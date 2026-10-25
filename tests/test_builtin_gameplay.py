"""Synthetic admission/rollback preflight, never native gameplay evidence."""
from copy import deepcopy
from contextlib import ExitStack
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    raise unittest.SkipTest("Optional torch is not installed") from error

from experiments.learner import builtin_gameplay as gameplay
from experiments.learner.gameplay import assert_same_state
from experiments.learner.ppo import PPOConfig
from experiments.learner import contextual as c, contextual_model as m
from experiments.learner.builtin_rollout import BuiltinCollectedEpisode, _opponent_token
from experiments.learner.contextual_rollout import RecurrentPrefix, _frame_bytes
from experiments.headless import builtin_evidence as evidence
from tests.test_builtin_env import manifest_for
from tests.test_builtin_ppo_numerics import synthetic_dialog_rollout


def synthetic_experience(model, *, player=0, boundary="natural", closure=False, episode=1, seed=11):
    """Typed private records only; these fixtures make no native-proof claim."""
    prior = synthetic_dialog_rollout(model, player=player, boundary=boundary, terminal_closure=closure)
    history = c.PlayerHistory(str(episode), player, schema=c.DIALOGS_SCHEMA)
    prefixes, hasher = {}, hashlib.sha256()
    for frame in prior.history:
        if frame.kind in ("finalized", "cutoff_finalized"):
            continue
        frame = replace(frame, episode=str(episode))
        history.append(frame)
        hasher.update(_frame_bytes(frame))
        prefixes[frame.seq] = hasher.hexdigest()
    manifest = manifest_for(history, {"natural": "completed", "cutoff": "truncated", "failure": "failed"}[boundary])
    manifest.update(seed=seed, identity={"synthetic": True, "native_runtime_sha256": {"test": "a"*64}})
    weight = m.weights_sha256(model)
    def prefix(seq): return RecurrentPrefix(str(episode), player, "synthetic-fixed", weight, seq, prefixes[seq])
    opponent = manifest["opponent_identity"]
    choices = tuple(replace(item, prefix=prefix(item.decision_seq), opponent_checkpoint=_opponent_token(opponent, player))
                    for item in prior.decisions)
    estimates = tuple(replace(item, prefix=prefix(item.decision_seq)) for item in prior.estimates)
    return BuiltinCollectedEpisode(history.frames, choices, estimates, "synthetic-fixed", weight, 17+episode-1,
        json.dumps(opponent), json.dumps(manifest), json.dumps(model.config.to_dict()))


def write_plan(root, *, rows=4, allow_prefix=True, native_identity=None, runtime=None):
    checkpoint = root / "synthetic-checkpoint.pt"
    checkpoint.write_bytes(b"synthetic fixture; not a model checkpoint")
    implementation_name = "experiments/learner/builtin_gameplay.py"
    implementation_path = Path(gameplay.__file__).resolve()
    plan = {"schema": "heroes-builtin-ppo-pilot-v1-plan", "source": {"synthetic": True},
        "native": native_identity or {"synthetic": True}, "runtime": runtime or {"test": "a"*64},
        "implementation": {implementation_name: gameplay.digest(implementation_path)},
        "checkpoint": {"path": str(checkpoint), **evidence.native.file_identity(checkpoint)},
        "configuration": {"max_days": 30, "ordinary_choice_limit": 248, "max_total_choices": 256,
            "allow_verified_training_prefix": allow_prefix, "ppo": PPOConfig(epochs=1).to_dict()},
        "schedule": [{"episode": i+1, "phase": "training", "engine_seed": 11+i,
                      "learner_player": i % 2, "policy_seed": 17+i} for i in range(rows)]}
    path = root / "plan.json"
    path.write_text(json.dumps(plan))
    return path, plan


def binding_for(root, model, plan_path, plan, index):
    row = plan["schedule"][index]
    return gameplay.TrainingBinding.create(plan_path=plan_path, plan_sha256=gameplay.digest(plan_path),
        schedule_index=index, folder=root / f"episode-{row['episode']:06d}", checkpoint="synthetic-fixed",
        checkpoint_file=plan["checkpoint"]["path"], weights_sha256=m.weights_sha256(model),
        opponent_identity=evidence.opponent_identity(row["learner_player"], "a"*64), source_identity=plan["source"],
        native_identity=plan["native"], max_days=30)


class AdmissionContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def model(self):
        return m.ContextualActorCritic(m.ModelConfig(schema=m.DIALOGS_MODEL_SCHEMA, hidden_size=8, feature_size=4))

    def receipt(self, root, model, plan_path, plan, index):
        bound = binding_for(root, model, plan_path, plan, index)
        folder = Path(bound.folder); folder.mkdir()
        item = synthetic_experience(model, player=bound.row["learner_player"], episode=bound.row["episode"],
                                    seed=bound.row["engine_seed"])
        reference = "synthetic-only-external-proof"
        rollout = gameplay.BuiltinTrainingRollout(item, bound, reference, gameplay._samples(item, reference))
        watched = folder / "synthetic-evidence.json"
        watched.write_text("{}")
        receipt = gameplay.VerifiedTrainingReceipt(bound, rollout,
            json.dumps({watched.name: evidence.native.file_identity(watched)}), str(root / "synthetic-verifier"), "{}", "{}")
        (folder / "training-receipt.json").write_text(json.dumps(receipt.to_dict()))
        # Explicit synthetic injection exercises this private capability gate;
        # production receipts can only be minted after the real verifier.
        gameplay._RECEIPTS[id(receipt)] = receipt
        return receipt

    def test_targets_both_seats_cutoff_and_natural_without_opponent_frames(self):
        model = self.model()
        for player in (0, 1):
            for boundary, closure in (("natural", False), ("natural", True), ("cutoff", False)):
                with self.subTest(player=player, boundary=boundary, closure=closure):
                    item = synthetic_experience(model, player=player, boundary=boundary, closure=closure)
                    before = tuple(c.encode_frame(frame) for frame in item.history)
                    samples = gameplay._samples(item, "synthetic-proof")
                    self.assertEqual(len(samples), 2)
                    self.assertEqual([sample.reward for sample in samples], [0, (1 if player == 0 else -1) if boundary == "natural" else 0])
                    self.assertEqual(samples[0].successor_seq, samples[1].decision.decision_seq)
                    if boundary == "cutoff":
                        self.assertEqual(samples[-1].successor_seq, item.history[-1].decision_seq)
                        self.assertIsNotNone(samples[-1].bootstrap_value)
                    else:
                        self.assertIsNone(samples[-1].successor_seq)
                        self.assertIsNone(samples[-1].bootstrap_value)
                    self.assertEqual(samples[-1].terminal_closure is not None, closure)
                    self.assertTrue(all(frame.player == player for frame in item.history))
                    self.assertFalse(any(frame.kind in ("finalized", "cutoff_finalized") for frame in item.history))
                    self.assertEqual(tuple(c.encode_frame(frame) for frame in item.history), before)

    def test_failure_and_missing_consumed_result_cannot_supply_targets(self):
        model = self.model()
        item = synthetic_experience(model, boundary="failure")
        with self.assertRaisesRegex(ValueError, "Technical failures"):
            gameplay._samples(item, "synthetic")
        item = synthetic_experience(model, boundary="natural")
        with self.assertRaises(ValueError):
            replace(item, history=tuple(f for f in item.history if f.kind != "result"))

    def test_unacted_terminal_supersession_does_not_steal_last_consumed_reward(self):
        item = synthetic_experience(self.model(), boundary="cutoff")
        history = c.PlayerHistory(item.history[0].episode, item.player, schema=c.DIALOGS_SCHEMA)
        for frame in item.history[:-1]:
            history.append(frame)
        held = history.decision.seq
        history.emit("superseded", decision_seq=held, reason="terminal")
        history.emit("terminal", outcome="loss")
        ended = replace(item, history=history.frames, manifest_json=json.dumps(manifest_for(history, "completed")))
        samples = gameplay._samples(ended, "synthetic-proof")
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[-1].reward, -1)
        self.assertNotEqual(samples[-1].decision.decision_seq, held)
        self.assertIsNone(samples[-1].terminal_closure)
        self.assertIsNone(samples[-1].bootstrap_value)

    def test_explicit_nonempty_prefix_and_all_four_admit_but_gaps_and_reordering_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, model = Path(tmp).resolve(), self.model()
            path, plan = write_plan(root)
            receipts = tuple(self.receipt(root, model, path, plan, i) for i in range(4))
            with patch("experiments.headless.builtin_replay.verify_identity"):
                for count in (1, 2, 3, 4):
                    got = gameplay._admit_builtin(receipts[:count], model, plan_sha256=gameplay.digest(path))
                    self.assertEqual(len(got), count)
                for changed in ((receipts[1],), (receipts[0], receipts[2]), receipts[::-1], (receipts[0], receipts[0]), ()):
                    with self.assertRaises(ValueError):
                        gameplay._admit_builtin(changed, model, plan_sha256=gameplay.digest(path))

    def test_short_prefix_requires_explicit_predeclared_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, model = Path(tmp).resolve(), self.model()
            path, plan = write_plan(root, allow_prefix=False)
            receipt = self.receipt(root, model, path, plan, 0)
            with self.assertRaisesRegex(ValueError, "contiguous"):
                gameplay._admit_builtin((receipt,), model, plan_sha256=gameplay.digest(path))

    def test_receipt_replacement_diagnostic_data_and_evidence_mutation_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, model = Path(tmp).resolve(), self.model()
            path, plan = write_plan(root, rows=1)
            receipt = self.receipt(root, model, path, plan, 0)
            for invalid in (replace(receipt), receipt.to_dict(), receipt.rollout.experience):
                with self.assertRaises(ValueError):
                    gameplay._admit_builtin((invalid,), model, plan_sha256=gameplay.digest(path))
            watched = Path(receipt.binding.folder) / "synthetic-evidence.json"
            watched.write_text('{"tampered":true}')
            with patch("experiments.headless.builtin_replay.verify_identity"), self.assertRaisesRegex(ValueError, "evidence changed"):
                gameplay._admit_builtin((receipt,), model, plan_sha256=gameplay.digest(path))

    def test_old_or_relabelled_capture_cannot_request_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, model = Path(tmp).resolve(), self.model()
            path, plan = write_plan(root, rows=1)
            bound = binding_for(root, model, path, plan, 0)
            capture = gameplay.TrainingCapture(synthetic_experience(model), bound)
            for invalid in (capture, capture.experience, capture.to_dict()):
                with self.assertRaisesRegex(ValueError, "fresh training collector"):
                    gameplay.verify_training_capture(bound.folder, invalid, bound, model,
                        replay_binary=root / "unused", data=root, remaining=lambda: 1.0)

    def test_plan_checkpoint_and_binding_drift_reject_before_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, model = Path(tmp).resolve(), self.model()
            path, plan = write_plan(root, rows=1)
            bound = binding_for(root, model, path, plan, 0)
            with self.assertRaisesRegex(ValueError, "prelaunch binding"):
                gameplay.TrainingCollector(model, replace(bound))
            Path(plan["checkpoint"]["path"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                bound.recheck()

    def test_receipts_are_single_use_even_after_rolled_back_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, model = Path(tmp).resolve(), self.model()
            path, plan = write_plan(root, rows=1)
            receipt = self.receipt(root, model, path, plan, 0)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
            with patch("experiments.headless.builtin_replay.verify_identity"), patch.object(gameplay, "transactional_update", side_effect=TimeoutError("synthetic")):
                with self.assertRaises(TimeoutError):
                    gameplay.native_update(model, optimizer, (receipt,), plan_sha256=gameplay.digest(path),
                        config=PPOConfig(epochs=1), generator=torch.Generator(), remaining=lambda: 1.0)
            with self.assertRaisesRegex(ValueError, "already used"):
                gameplay._admit_builtin((receipt,), model, plan_sha256=gameplay.digest(path))


class FreshTrainingIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def collect(self, root, player, boundary):
        from tests.test_builtin_rollout import SyntheticLauncher
        from experiments.headless import builtin_env as be
        from experiments.headless.env import Limits
        launcher = SyntheticLauncher(root, boundary)
        launcher.identity = lambda: {"synthetic": True}
        model = m.ContextualActorCritic(m.ModelConfig(schema=m.DIALOGS_MODEL_SCHEMA, hidden_size=8, feature_size=4))
        model.eval().requires_grad_(False)
        path, plan = write_plan(root, rows=1, runtime=launcher.runtime)
        plan["schedule"][0]["learner_player"] = player
        path.write_text(json.dumps(plan))
        opponent = evidence.opponent_identity(player, launcher.runtime["AI/libNullkiller.dylib"])
        bound = gameplay.TrainingBinding.create(plan_path=path, plan_sha256=gameplay.digest(path), schedule_index=0,
            folder=root / "episode-000001", checkpoint="synthetic-fixed", checkpoint_file=plan["checkpoint"]["path"],
            weights_sha256=m.weights_sha256(model), opponent_identity=opponent, source_identity=plan["source"],
            native_identity=plan["native"], max_days=30)
        collector = gameplay.TrainingCollector(model, bound)
        env = be.BuiltinHeadlessEnv(launcher, root, learner_player=player, limits=Limits(max_actions=1))
        identity = {"synthetic": True, "engine_revision": "synthetic-not-native", "map_sha256": "b"*64,
                    "native_runtime_sha256": launcher.runtime}
        with ExitStack() as stack:
            stack.enter_context(patch.object(env, "_identity", return_value=identity))
            stack.enter_context(patch.object(env, "_engine_evidence", return_value=({"synthetic": True}, None)))
            stack.enter_context(patch.object(evidence, "capture_files", return_value={"schema": evidence.REPLAY_SCHEMA,"recorded":True,"files":{}}))
            stack.enter_context(patch.object(be.native, "terminal_capture_close"))
            stack.enter_context(patch.object(be, "stop_process_group", side_effect=lambda p: p.stop()))
            try:
                captured = collector.run_episode(env, episode=1, seed=11)
            finally:
                env.close()
        launcher.thread.join(2)
        self.assertFalse(launcher.thread.is_alive())
        self.assertIsNone(launcher.process.error)
        folder = env.episode_dir
        (folder / "engine").mkdir(exist_ok=True)
        (folder / "engine/experiment-metadata.json").write_text('{"synthetic":true}')
        return captured, bound, model, folder

    def verify(self, captured, bound, model, folder, *, diagnostic=False, mutate=False):
        from experiments.headless import builtin_replay as replay
        from experiments.headless.builtin_replay_history import verify_history
        def synthetic_native(episode, *, output, **kwargs):
            # Only native process execution/effect verification is mocked.
            # Own transport reconstruction and fixed-model resampling are real.
            output.mkdir()
            final = evidence.read_finalization(episode, captured.manifest)
            history = verify_history(episode, captured.manifest, final)
            result = {"schema": replay.DIAGNOSTIC_SCHEMA if diagnostic else replay.REPORT_SCHEMA,
                "verified": not diagnostic, "identity_check": {"mode": "diagnostic_verifier_only" if diagnostic else "exact_capture"},
                "history": history, "wire": {}, "native": {"synthetic": True, "returncode": 0, "seconds": 0.0}}
            (output / "native-report.json").write_text('{"synthetic":true}')
            (output / "builtin-replay-verification.json").write_text(json.dumps(result))
            if mutate:
                (episode / "engine/experiment-metadata.json").write_text('{"tampered":true}')
            return result
        with patch.object(replay, "verify_episode", side_effect=synthetic_native), patch.object(replay, "validate_native_report") as native_check:
            receipt = gameplay.verify_training_capture(folder, captured, bound, model, replay_binary=folder / "unused-verifier",
                data=folder, remaining=lambda: 30.0)
            native_check.assert_called_once()
            return receipt

    def test_fresh_collect_parse_history_resample_and_receipt_both_seats(self):
        for player in (0, 1):
            for boundary in ("natural", "cutoff"):
                with self.subTest(player=player, boundary=boundary), tempfile.TemporaryDirectory() as temp:
                    captured, bound, model, folder = self.collect(Path(temp).resolve(), player, boundary)
                    self.assertEqual(len(captured.decisions), 1)
                    self.assertFalse((folder / "builtin-rollout.json").exists())
                    self.assertFalse(captured.experience.optimization_eligible)
                    receipt = self.verify(captured, bound, model, folder)
                    self.assertEqual(len(receipt.rollout.samples), 1)
                    self.assertEqual(receipt.rollout.boundary, boundary)
                    self.assertEqual(json.loads(receipt.policy_reconstruction_json)["actions_resampled"], 1)
                    with patch("experiments.headless.builtin_replay.verify_identity"):
                        admitted = gameplay._admit_builtin((receipt,), model, plan_sha256=bound.plan_sha256)
                    self.assertEqual(admitted, (receipt.rollout,))

    def test_diagnostic_native_replay_and_mid_verification_mutation_do_not_mint_receipt(self):
        for diagnostic, mutate in ((True, False), (False, True)):
            with self.subTest(diagnostic=diagnostic, mutate=mutate), tempfile.TemporaryDirectory() as temp:
                captured, bound, model, folder = self.collect(Path(temp).resolve(), 0, "natural")
                with self.assertRaises(ValueError):
                    self.verify(captured, bound, model, folder, diagnostic=diagnostic, mutate=mutate)
                self.assertFalse((folder / "training-receipt.json").exists())


class TransactionRollbackTests(unittest.TestCase):
    def test_exception_after_actual_adam_step_restores_all_state(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
        model.eval()
        model[1].train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        parameters = tuple(model.parameters())
        parameters[-1].requires_grad_(False)
        generator = torch.Generator().manual_seed(139)
        state, adam = deepcopy(model.state_dict()), deepcopy(optimizer.state_dict())
        gradients = tuple(p.grad.clone() for p in parameters)
        flags = tuple(p.requires_grad for p in parameters)
        modes = tuple(m.training for m in model.modules())
        rng, python_rng, batch_rng = torch.get_rng_state().clone(), random.getstate(), generator.get_state().clone()

        def interrupted(actual_model, actual_optimizer, rollouts, **kwargs):
            actual_model.train()
            actual_model.requires_grad_(True)
            actual_optimizer.zero_grad(set_to_none=True)
            actual_model(torch.ones(1, 2)).sum().backward()
            actual_optimizer.step()
            kwargs["after_step"]()
            torch.rand(2)
            random.random()
            torch.rand(2, generator=kwargs["generator"])
            raise TimeoutError("synthetic deadline after Adam")

        with patch.object(gameplay, "_update", side_effect=interrupted), self.assertRaises(gameplay.UpdateRolledBack) as caught:
            gameplay.transactional_update(model, optimizer, (), config=PPOConfig(epochs=1),
                generator=generator, remaining=lambda: 1.0, max_optimizer_steps=32)
        self.assertEqual(caught.exception.completed_steps, 1)
        assert_same_state(model.state_dict(), state)
        assert_same_state(optimizer.state_dict(), adam)
        self.assertEqual(tuple(p.requires_grad for p in parameters), flags)
        self.assertEqual(tuple(m.training for m in model.modules()), modes)
        for parameter, gradient in zip(parameters, gradients):
            self.assertTrue(torch.equal(parameter.grad, gradient))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertEqual(random.getstate(), python_rng)
        self.assertTrue(torch.equal(generator.get_state(), batch_rng))

    def test_deadline_after_kernel_returns_rolls_back_completed_update(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
        before = deepcopy(model.state_dict())
        calls = 0

        def remaining():
            nonlocal calls
            calls += 1
            if calls == 2:
                raise TimeoutError("synthetic expiry before commit")
            return 1.0

        def update(actual_model, actual_optimizer, rollouts, **kwargs):
            actual_model(torch.ones(1, 1)).sum().backward()
            actual_optimizer.step()
            kwargs["after_step"]()
            return object()

        with patch.object(gameplay, "_update", side_effect=update), self.assertRaises(gameplay.UpdateRolledBack):
            gameplay.transactional_update(model, optimizer, (), config=PPOConfig(epochs=1),
                generator=torch.Generator().manual_seed(2), remaining=remaining, max_optimizer_steps=32)
        assert_same_state(model.state_dict(), before)
        self.assertFalse(optimizer.state)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
