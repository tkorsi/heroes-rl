"""Owned exec replay transport and coordinator admission; no native gameplay."""
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.headless import contextual_evidence as evidence, process_deadline, spawned_replay
from experiments.learner import builtin_gameplay as gameplay, parallel_collection as parallel
from tests import test_builtin_gameplay as fixtures


class SpawnedReplayTransportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.jobs = []
        self.addCleanup(self.close_jobs)

    def close_jobs(self):
        for job in self.jobs:
            job.release()
        for job in self.jobs:
            job.stop(deadline_monotonic=time.monotonic()+8, deadline_wall=time.time()+8)

    def job(self, name="episode", seconds=20):
        folder = self.root / name
        folder.mkdir()
        (folder / "capture.json").write_text('{"fixed":true}')
        job = spawned_replay.SpawnedReplay(folder, binary=self.root / "unused-native", data=self.root,
            original_files={"capture.json": evidence.file_identity(folder / "capture.json")},
            deadline_monotonic=time.monotonic()+seconds, deadline_wall=time.time()+seconds)
        self.jobs.append(job)
        return job

    def launch(self, job, extra="", *, token="r['token']"):
        # Harmless fresh interpreters exercise the actual deadline guardian and
        # pipe handshake. They make no native verification or training claim.
        code = """
import hashlib, json, pathlib, sys, time
r = json.loads(sys.stdin.buffer.readline())
folder = pathlib.Path(r['folder'])
""" + extra + """
out = folder / 'training-verification'
out.mkdir()
path = out / 'builtin-replay-verification.json'
path.write_text(json.dumps({'synthetic': True, 'episode': folder.name}))
raw = path.read_bytes()
""" + f"print(json.dumps({{'schema': r['schema'], 'token': {token}, 'report': {{'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}}}}), flush=True)\n"
        actual = process_deadline.guarded_popen
        def spawn(command, **kwargs):
            return actual([sys.executable, "-c", code], **kwargs)
        with patch.object(spawned_replay, "guarded_popen", side_effect=spawn):
            job.start()

    def test_four_exec_children_overlap_and_results_retain_requested_order(self):
        jobs = [self.job(str(i)) for i in range(4)]
        extra = """
(folder / 'ready').touch()
while sum((folder.parent / str(i) / 'ready').exists() for i in range(4)) < 4:
    time.sleep(.01)
time.sleep(.04 * (3-int(folder.name)))
"""
        for job in jobs:
            self.launch(job, extra)
        results = [job.result(timeout=15) for job in jobs]
        self.assertEqual([result["episode"] for result in results], [str(i) for i in range(4)])
        for job in jobs:
            job.stop()
            self.assertFalse(process_deadline.group_live_members(job.process.pid))
        with self.assertRaisesRegex(ValueError, "not fresh"):
            jobs[0].result(timeout=1)

    def test_forged_pipe_token_is_rejected(self):
        job = self.job()
        self.launch(job, token="'f'*64")
        with self.assertRaisesRegex(ValueError, "live launch"):
            job.result(timeout=10)

    def test_capture_and_report_tampering_are_rejected(self):
        for kind in ("capture", "report"):
            with self.subTest(kind=kind):
                job = self.job(kind)
                self.launch(job)
                job.process.wait(timeout=10)
                path = job.folder / "capture.json" if kind == "capture" else job.output / "builtin-replay-verification.json"
                path.write_text('{"tampered":true}')
                with self.assertRaisesRegex(ValueError, "changed"):
                    job.result(timeout=10)

    def test_completion_parses_only_the_report_bytes_bound_by_its_private_digest(self):
        job = self.job()
        self.launch(job)
        job.process.wait(timeout=10)
        report = job.output / "builtin-replay-verification.json"
        original_identity = evidence.file_identity
        def mutation_after_hash(path):
            result = original_identity(path)
            if Path(path) == report:
                report.write_text('{"synthetic":true,"episode":"unbound replacement"}')
            return result
        # A separate hash-then-open implementation accepts the replacement.
        # Binding and parsing one byte snapshot keeps the child's actual result.
        with patch.object(evidence, "file_identity", side_effect=mutation_after_hash):
            result = job.result(timeout=10)
        self.assertEqual(result["episode"], "episode")

    def test_worker_will_not_attest_a_persisted_report_different_from_execution(self):
        from experiments.headless import background_check, builtin_replay
        job = self.job()
        request = dict(schema=spawned_replay.SCHEMA, token=job.token, folder=str(job.folder),
                       binary=str(job.binary), data=str(job.data), original_files=job.original_files,
                       deadline_monotonic=job.deadline, deadline_wall=job.wall_deadline)
        def altered(*args, output, **kwargs):
            output.mkdir()
            (output / "builtin-replay-verification.json").write_text('{"verified":true}')
            return {"verified": False, "error": "actual replay rejection"}
        output = io.StringIO()
        with patch.object(spawned_replay.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(json.dumps(request).encode()+b"\n"))), \
                patch.object(builtin_replay, "verify_episode", side_effect=altered), \
                patch.object(background_check, "FORK", False), redirect_stdout(output), \
                self.assertRaisesRegex(ValueError, "differs from its executed verification"):
            spawned_replay._worker()
        self.assertEqual(output.getvalue(), "")

    def test_failure_cancellation_reaps_owned_descendants_only(self):
        job = self.job()
        unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
        try:
            self.launch(job, """
import subprocess
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
(folder / 'descendant').write_text(str(child.pid))
time.sleep(30)
""")
            end = time.monotonic()+10
            while not (job.folder / "descendant").exists():
                self.assertLess(time.monotonic(), end)
                time.sleep(.01)
            job.stop()
            self.assertFalse(process_deadline.group_live_members(job.process.pid))
            self.assertIsNone(unrelated.poll())
        finally:
            unrelated.terminate()
            unrelated.wait(timeout=5)

    def test_existing_output_or_changed_capture_cannot_be_launched(self):
        for kind in ("existing", "changed"):
            job = self.job(kind)
            if kind == "existing":
                job.output.mkdir()
            else:
                (job.folder / "capture.json").write_text("changed")
            with self.assertRaises(ValueError):
                job.start()
            self.assertIsNone(job.process)


class PreparedVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.FreshTrainingIntegrationTests.setUpClass()

    @classmethod
    def tearDownClass(cls):
        fixtures.FreshTrainingIntegrationTests.tearDownClass()

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        fixture = fixtures.FreshTrainingIntegrationTests()
        self.capture, self.binding, self.model, self.folder = fixture.collect(Path(temporary.name).resolve(), 0, "natural")

    def prepare(self, stack):
        from experiments.headless import builtin_replay as replay, builtin_replay_history
        captured, folder = self.capture, self.folder
        class SyntheticReplay:
            def __init__(self, folder, *, binary, data, **kwargs):
                self.binary, self.data = Path(binary).resolve(), Path(data).resolve()
                self.stopped = False
            def start(self): return self
            def result(self, *, timeout):
                output = folder / "training-verification"
                output.mkdir()
                final = gameplay.evidence.read_finalization(folder, captured.manifest)
                history = builtin_replay_history.verify_history(folder, captured.manifest, final)
                report = {"schema": replay.REPORT_SCHEMA, "verified": True, "identity_check": {"mode": "exact_capture"},
                          "history": history, "wire": {}, "native": {"synthetic": True, "returncode": 0, "seconds": 0.}}
                (output / "native-report.json").write_text('{"synthetic":true}')
                (output / "builtin-replay-verification.json").write_text(json.dumps(report))
                return report
            def stop(self): self.stopped = True
        stack.enter_context(patch.object(spawned_replay, "SpawnedReplay", SyntheticReplay))
        stack.enter_context(patch.object(replay, "validate_native_report"))
        prepared = gameplay.start_training_verification(folder, self.capture, self.binding, self.model,
            replay_binary=folder / "unused", data=folder, remaining=lambda: 30.,
            deadline_monotonic=time.monotonic()+30, deadline_wall=time.time()+30)
        self.addCleanup(gameplay.discard_training_verification, prepared)
        return prepared

    def verify(self, prepared):
        return gameplay.verify_training_capture(self.folder, self.capture, self.binding, self.model,
            replay_binary=self.folder / "unused", data=self.folder, remaining=lambda: 30., prepared=prepared)

    def test_prepared_child_replay_retains_real_coordinator_policy_reconstruction(self):
        with ExitStack() as stack:
            prepared = self.prepare(stack)
            self.assertFalse((self.folder / "training-receipt.json").exists())
            receipt = self.verify(prepared)
            report = json.loads(receipt.policy_reconstruction_json)
            self.assertEqual(report["actions_resampled"], len(self.capture.decisions))
            self.assertEqual(report["maximum_absolute_difference"], 0)
            self.assertTrue(prepared.job.stopped)
            self.assertNotIn(id(prepared), gameplay._PREPARED_VERIFICATIONS)
            with self.assertRaisesRegex(ValueError, "unchanged fresh capture"):
                self.verify(prepared)

    def test_modified_capture_cannot_use_an_earlier_prepared_replay(self):
        with ExitStack() as stack:
            prepared = self.prepare(stack)
            (self.folder / "engine/experiment-metadata.json").write_text('{"tampered":true}')
            with self.assertRaisesRegex(ValueError, "unchanged fresh capture"):
                self.verify(prepared)
            self.assertFalse((self.folder / "training-receipt.json").exists())

    def test_serialized_or_unregistered_preparation_cannot_mint_receipt(self):
        with ExitStack() as stack:
            prepared = self.prepare(stack)
            forged = SimpleNamespace(**vars(prepared))
            with self.assertRaisesRegex(ValueError, "unchanged fresh capture"):
                self.verify(forged)
            gameplay.discard_training_verification(prepared)
            with self.assertRaisesRegex(ValueError, "unchanged fresh capture"):
                self.verify(prepared)
            self.assertFalse((self.folder / "training-receipt.json").exists())


class VerificationCleanupTests(unittest.TestCase):
    def test_releases_every_worker_before_waiting_and_reports_uncertain_cleanup(self):
        events = []
        class Job:
            def __init__(self, index): self.index = index
            def release(self): events.append(("release", self.index))
            def stop(self, **kwargs):
                events.append(("stop", self.index))
                if self.index == 0:
                    raise TimeoutError("synthetic expired cleanup")
        budget = SimpleNamespace(deadline=time.monotonic()+10, wall_deadline=time.time()+10)
        original = ValueError("original verification rejection")
        parallel._stop_verifications({i: Job(i) for i in range(4)}, {}, budget, original)
        self.assertEqual(events[:4], [("release", i) for i in range(4)])
        self.assertEqual(events[4:], [("stop", i) for i in range(4)])
        self.assertFalse(original.verification_cleanup_verified)
        self.assertIn("synthetic expired cleanup", original.__notes__[0])


class ParallelVerificationPipelineTests(unittest.TestCase):
    def run_pipeline(self, root, *, failure=None):
        from experiments.learner import builtin_training as runner
        # Import before temporarily replacing the model identity function.
        from validation.probes import validate_hero_policy
        rows = [{**row, "phase": "evaluation"} for row in runner.schedule(1)]
        folders = [root / f"episode-{row['episode']:06d}" for row in rows]
        plan = dict(schema="heroes-builtin-checkpoint-evaluation-v1-plan",
            configuration=dict(allow_verified_training_prefix=False, cleanup_reserve_seconds=0),
            execution=dict(collection_workers=4, verification_workers=4), execution_output=str(root),
            schedule=rows, runtime={}, data="/synthetic/data", replay_binary={"path": "/synthetic/replay"})
        events, jobs = [], []
        class SyntheticReplay:
            def __init__(self, folder, **kwargs):
                self.index = folders.index(folder)
                jobs.append(self)
            def start(self): events.append(("start", self.index))
            def result(self, **kwargs):
                events.append(("result", self.index))
                if self.index == failure:
                    raise ValueError("synthetic spawned replay rejection")
                return {"verified": True}
            def stop(self, **kwargs): events.append(("stop", self.index))
            def release(self): events.append(("release", self.index))
        def supervise(*args, **kwargs):
            for folder in folders:
                folder.mkdir()
                (folder / "manifest.json").write_text("{}")
        def reconstruct(folder, *args):
            events.append(("reconstruct", folders.index(folder)))
            return {}
        with ExitStack() as stack:
            stack.enter_context(patch.object(parallel, "_read_plan", return_value=plan))
            stack.enter_context(patch.object(parallel, "_check_identity"))
            stack.enter_context(patch.object(parallel, "_supervise", side_effect=supervise))
            stack.enter_context(patch.object(gameplay, "_capture_file_bindings", return_value={}))
            stack.enter_context(patch.object(spawned_replay, "SpawnedReplay", SyntheticReplay))
            stack.enter_context(patch("experiments.learner.contextual_model.weights_sha256", return_value="fixed"))
            stack.enter_context(patch.object(runner.pilot, "native_opponent", return_value={}))
            stack.enter_context(patch("experiments.learner.builtin_rollout.audit_persisted_builtin_policy", side_effect=reconstruct))
            stack.enter_context(patch("experiments.learner.builtin_behavior.analyze_episode", return_value={}))
            no_serial = stack.enter_context(patch("experiments.headless.builtin_replay.verify_episode",
                                                  side_effect=AssertionError("unexpected serial replay")))
            try:
                result = parallel.collect_batch(root / "plan.json", "plan-hash", object(), "checkpoint",
                    workers=4, verification_workers=4,
                    deadline_monotonic=time.monotonic()+30, deadline_wall=time.time()+30)
            except parallel.CollectionFailed:
                if failure is None:
                    raise
                result = None
            no_serial.assert_not_called()
        return result, events, json.loads((root / "parallel-collection/result.json").read_text())

    def test_all_four_start_before_ordered_coordinator_checks(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, events, report = self.run_pipeline(Path(temporary))
        self.assertEqual(events[:4], [("start", i) for i in range(4)])
        self.assertEqual([event for event in events if event[0] == "result"], [("result", i) for i in range(4)])
        self.assertEqual([event for event in events if event[0] == "reconstruct"], [("reconstruct", i) for i in range(4)])
        self.assertEqual(len(result.attempts), 4)
        self.assertEqual(report["verification_workers"], 4)
        self.assertEqual(report["training_samples_admitted"], 0)

    def test_one_replay_rejection_cancels_siblings_and_revokes_whole_batch(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, events, report = self.run_pipeline(Path(temporary), failure=1)
        self.assertIsNone(result)
        self.assertEqual([event for event in events if event[0] == "release"], [("release", i) for i in range(4)])
        self.assertTrue(all(("stop", i) in events for i in range(4)))
        self.assertTrue(all(row["completion_receipt"] is False and row["batch_admission"] == "rejected"
                            for row in report["attempts"]))
        self.assertEqual((report["optimizer_updates"], report["training_samples_admitted"]), (0, 0))
        self.assertIn("synthetic spawned replay rejection", report["error"])


if __name__ == "__main__":
    unittest.main()
