"""Synthetic and file-only checks for the fixed-weight v5 validation runner."""
from dataclasses import replace
import json
from pathlib import Path
import signal
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    raise unittest.SkipTest("Optional torch dependency is not installed") from error

from experiments.explorer.create_hero_scenario import BASELINE, create_hero_scenario
from experiments.learner import contextual as c
from experiments.learner.contextual_model import ContextualActorCritic, ModelConfig, weights_sha256
from experiments.learner.contextual_rollout import ContextualCollector, ContextualSession, DecisionEstimate
from experiments.learner.inference import ordinary_map_identity
from tests.test_contextual_rollout import finish_pair, initial, request, resolve
from tests.test_contextual_heroes import history_at_tavern
from validation.probes import validate_hero_policy as probe

torch.set_num_threads(1)


class HeroPolicySettingsTests(unittest.TestCase):
    def test_fixed_pair_has_both_seats_and_distinct_player_rngs(self):
        settings = probe.Settings()
        rows = settings.schedule()
        self.assertEqual(len(rows), 2)
        self.assertEqual([row["learner_player"] for row in rows], [0, 1])
        self.assertEqual({row["engine_seed"] for row in rows}, {settings.engine_seed})
        seeds = [seed for row in rows for seed in row["policy_seeds"]]
        self.assertEqual(len(set(seeds)), 4)
        rows[0]["engine_seed"] = 999
        self.assertEqual(settings.schedule()[0]["engine_seed"], settings.engine_seed)

    def test_invalid_or_unbounded_settings_fail_before_native_launch(self):
        for name, value in (("engine_seed", 0), ("policy_seed", True), ("max_requests", 0),
                            ("max_requests", 513), ("max_days", 31), ("max_days", 1.0),
                            ("case_seconds", float("nan")), ("case_seconds", float("inf")),
                            ("case_seconds", 0), ("case_seconds", 181),
                            ("opponent_seed", probe.Settings().initialization_seed)):
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                replace(probe.Settings(), **{name: value})

    def test_exact_hero_scenario_is_separate_from_unchanged_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "heroes.vmap"
            create_hero_scenario(path)
            identity = probe.hero_map_identity(path)
            self.assertEqual(identity["scenario"], "hero-management-v1")
            self.assertFalse(identity["fixture"])
            with self.assertRaises(ValueError): ordinary_map_identity(path)
            with self.assertRaises(ValueError): probe.hero_map_identity(BASELINE)
            with zipfile.ZipFile(path) as archive:
                payload = {name: archive.read(name) for name in archive.namelist()}
            header = json.loads(payload["header.json"])
            header["heroLevelLimit"] = 2
            payload["header.json"] = json.dumps(header).encode()
            changed = Path(directory) / "changed.vmap"
            with zipfile.ZipFile(changed, "w") as archive:
                for name, value in payload.items(): archive.writestr(name, value)
            with self.assertRaises(ValueError): probe.hero_map_identity(changed)

    @unittest.skipUnless(hasattr(signal, "setitimer"), "POSIX case deadline")
    def test_wall_deadline_interrupts_and_restores_handler(self):
        previous = signal.getsignal(signal.SIGALRM)
        with self.assertRaises(TimeoutError):
            with probe.wall_limit(0.02): time.sleep(0.2)
        self.assertEqual(signal.getsignal(signal.SIGALRM), previous)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0.0, 0.0))


class HeroPolicyReconstructionTests(unittest.TestCase):
    def setUp(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(492)
            self.model = ContextualActorCritic(ModelConfig(schema=c_model_schema(), hidden_size=8, feature_size=4)).eval()
        self.seeds = (311, 987)
        collector = ContextualCollector((self.model, self.model), ("fixed-left", "fixed-right"), learner_player=0,
            opponent_pool="synthetic-fixed-pool", interface=c.HEROES_ENVIRONMENT,
            purpose="diagnostic", origin="synthetic", policy_seeds=self.seeds)
        collector.start("synthetic-collector")
        pair = tuple(initial(player, schema=c.HEROES_SCHEMA) for player in (0, 1))
        for history in pair: resolve(history, collector.act(request(history)))
        self.episode = collector.finish(finish_pair(pair, "cutoff"), histories=tuple(history.frames for history in pair))

    def reconstruct(self, rollout, *, decisions=None, estimates=None):
        return probe.reconstruct_behavior(rollout.history, rollout.decisions if decisions is None else decisions,
            rollout.estimates if estimates is None else estimates, self.model, self.seeds[rollout.player])

    def test_reconstructs_both_private_histories_and_unsampled_cutoff_values_exactly(self):
        before = weights_sha256(self.model)
        for rollout in self.episode.rollouts:
            result = self.reconstruct(rollout)
            self.assertEqual(result["actions_resampled"], 1)
            self.assertEqual(result["values_recomputed"], 2)
            self.assertEqual(result["maximum_absolute_difference"], 0)
            self.assertFalse(rollout.optimization_eligible)
        self.assertEqual(weights_sha256(self.model), before)

    def test_joint_conditional_and_value_tampering_is_detected(self):
        rollout = self.episode.rollouts[0]
        decision = rollout.decisions[0]
        conditional = replace(decision.conditionals[0], log_prob=decision.conditionals[0].log_prob - 0.125)
        changed = replace(decision, log_prob=decision.log_prob - 0.125,
                          conditionals=(conditional, *decision.conditionals[1:]))
        with self.assertRaisesRegex(ValueError, "probability changed"):
            self.reconstruct(rollout, decisions=(changed,))
        changed_value = replace(rollout.estimates[0], value=rollout.estimates[0].value + 0.25)
        with self.assertRaisesRegex(ValueError, "value is not exactly"):
            self.reconstruct(rollout, estimates=(changed_value, *rollout.estimates[1:]))

    def test_missing_reconstruction_target_and_changed_weights_are_rejected(self):
        rollout = self.episode.rollouts[0]
        extra = replace(rollout.estimates[0], decision_seq=999,
                        prefix=replace(rollout.estimates[0].prefix, through_seq=999))
        with self.assertRaisesRegex(ValueError, "Unreconstructed"):
            self.reconstruct(rollout, estimates=(*rollout.estimates, extra))
        with torch.no_grad(): next(self.model.parameters()).add_(0.5)
        with self.assertRaisesRegex(ValueError, "different fixed weights"):
            self.reconstruct(rollout)

    def test_checkpoint_roundtrip_rng_and_frozen_fingerprints(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixed.pt"
            rng = torch.get_rng_state().clone()
            loaded = probe.prepare_policy(path, probe.Settings(), {"purpose": "synthetic-file-test"})
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            identity = probe.frozen_fingerprints(loaded, path)
            self.assertEqual(identity, probe.frozen_fingerprints(loaded, path))
            with torch.no_grad(): next(loaded.model.parameters()).add_(0.25)
            with self.assertRaisesRegex(ValueError, "weights changed"):
                probe.frozen_fingerprints(loaded, path)

    def test_multi_choice_tavern_action_requires_its_exact_player_rng(self):
        history = history_at_tavern()
        session = ContextualSession(self.model, history.episode, history.player, "fixed", "opponent",
                                    schema=c.HEROES_SCHEMA, seed=981)
        session.consume(history.frames)
        decision = session.choose(history.decision.seq)
        estimates = tuple(DecisionEstimate(seq, value, session._prefixes[seq])
                          for seq, value in session._values.items())
        result = probe.reconstruct_behavior(history.frames, (decision,), estimates, self.model, 981)
        self.assertEqual(result["actions_resampled"], 1)
        self.assertLess(decision.log_prob, 0)
        # Wrong RNGs can coincidentally choose the same action, but cannot all
        # reproduce a non-degenerate distribution. Detect a differing choice.
        detected = False
        for seed in range(10):
            try:
                probe.reconstruct_behavior(history.frames, (decision,), estimates, self.model, seed)
            except ValueError as error:
                self.assertIn("Saved action is not reproduced", str(error))
                detected = True
                break
        self.assertTrue(detected)


class HeroPolicyFailureTests(unittest.TestCase):
    def environment(self, folder):
        end = SimpleNamespace(status="failed", termination_reason="watchdog_timeout",
                              outcomes=dict.fromkeys(probe.COLORS), rewards=dict.fromkeys(probe.COLORS))
        return SimpleNamespace(episode_dir=folder, process=object(), result=None,
                               _note_problem=Mock(), _finish=Mock(return_value=end), close=Mock())

    def attempt(self):
        return {"status": "failed", "error": "TimeoutError: original deadline",
                "outcomes": dict.fromkeys(probe.COLORS), "rewards": dict.fromkeys(probe.COLORS)}

    def test_external_deadline_finalizes_failure_and_preserves_partial_history(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            env, attempt = self.environment(folder), self.attempt()
            history = initial(0, schema=c.HEROES_SCHEMA)
            collector = SimpleNamespace(sessions=(SimpleNamespace(history=history, _choices={}),))
            probe.retain_failed_evidence(env, collector, attempt, TimeoutError("original deadline"))
            env._note_problem.assert_called_once_with(
                "watchdog_timeout", "Fixed-weight diagnostic failed: TimeoutError: original deadline")
            env._finish.assert_called_once_with(None)
            self.assertEqual(attempt["error"], "TimeoutError: original deadline")
            self.assertEqual(attempt["episode_status"], "failed")
            self.assertEqual(attempt["failure_finalization"], "persisted")
            partial = json.loads((folder / "partial-policy-evidence.json").read_text())
            self.assertTrue(partial["incomplete"])
            self.assertFalse(partial["training_input"])
            self.assertEqual(partial["players"][0]["history"], [c.encode_frame(frame) for frame in history.frames])

    def test_secondary_finalization_and_evidence_errors_do_not_replace_original(self):
        with tempfile.TemporaryDirectory() as directory:
            env, attempt = self.environment(Path(directory)), self.attempt()
            env._finish.side_effect = ValueError("broken native evidence")
            collector = SimpleNamespace(sessions=())
            with patch.object(probe, "write_json", side_effect=OSError("disk write failed")):
                probe.retain_failed_evidence(env, collector, attempt, TimeoutError("original deadline"))
            self.assertEqual(attempt["error"], "TimeoutError: original deadline")
            self.assertEqual(attempt["failure_finalization_error"], "ValueError: broken native evidence")
            self.assertEqual(attempt["partial_evidence_error"], "OSError: disk write failed")
            self.assertEqual(attempt["outcomes"], dict.fromkeys(probe.COLORS))
            self.assertEqual(attempt["rewards"], dict.fromkeys(probe.COLORS))

    @unittest.skipUnless(hasattr(signal, "setitimer"), "POSIX cleanup deadline")
    def test_failed_finalization_has_its_own_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            env, attempt = self.environment(Path(directory)), self.attempt()
            env._finish.side_effect = lambda _: time.sleep(0.2)
            probe.retain_failed_evidence(env, None, attempt, TimeoutError("original deadline"), cleanup_seconds=0.02)
            self.assertTrue(attempt["failure_finalization_error"].startswith("TimeoutError:"))
            self.assertEqual(attempt["error"], "TimeoutError: original deadline")
            self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0.0, 0.0))

    def test_later_audit_failure_does_not_rewrite_finalized_native_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            manifest = folder / "manifest.json"
            manifest.write_text('{"status":"truncated"}\n')
            before = manifest.read_bytes()
            env, attempt = self.environment(folder), self.attempt()
            env.result = object()
            probe.retain_failed_evidence(env, None, attempt, ValueError("replay mismatch"))
            env._note_problem.assert_not_called()
            env._finish.assert_not_called()
            self.assertEqual(manifest.read_bytes(), before)

    def test_runner_stops_pair_after_failed_collection_and_always_closes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("map", "binary", "replay_binary"):
                (root / name).write_bytes(name.encode())
            (root / "data").mkdir()
            args = SimpleNamespace(**{**probe.asdict(probe.Settings()),
                **{name: root / name for name in ("map", "binary", "replay_binary", "data", "output")}})
            env = self.environment(None)
            def fail_collection(collector, current, *, episode, seed, max_requests):
                self.assertIs(current, env)
                env.episode_dir = args.output / f"episode-{episode:06d}"
                env.episode_dir.mkdir()
                raise TimeoutError("original deadline")
            launcher = SimpleNamespace(preflight=Mock(), identity=lambda: {"native": "synthetic"})
            with patch.object(probe, "VcmiLauncher", return_value=launcher), \
                 patch.object(probe, "hero_map_identity", return_value={"sha256": "synthetic"}), \
                 patch.object(probe, "runtime_identity", return_value={"runtime": "synthetic"}), \
                 patch.object(probe, "source_identity", return_value={"source": "synthetic"}), \
                 patch.object(probe, "implementation_identity", return_value={}), \
                 patch.object(probe.subprocess, "run"), \
                 patch.object(probe, "HeadlessSelfPlayEnv", return_value=env) as environment, \
                 patch.object(probe.ContextualCollector, "run_episode", autospec=True, side_effect=fail_collection):
                report = probe.run(args)
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["not_started"], 1)
            self.assertEqual(len(report["attempts"]), 1)
            self.assertEqual(report["attempts"][0]["error"], "TimeoutError: original deadline")
            self.assertEqual(report["attempts"][0]["failure_finalization"], "persisted")
            self.assertEqual(report["attempts"][0]["rewards"], dict.fromkeys(probe.COLORS))
            self.assertEqual(report["optimizer_steps"], 0)
            self.assertEqual(json.loads((args.output / "report.json").read_text()), report)
            environment.assert_called_once()
            env.close.assert_called_once()


def c_model_schema():
    return probe.model_schema_for(c.HEROES_ENVIRONMENT)


if __name__ == "__main__":
    unittest.main()
