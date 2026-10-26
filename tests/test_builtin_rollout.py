"""Socket-backed synthetic integration; no engine, replay or training claim.

The launcher emits a tiny valid native-shaped transport. Native proof producers
are explicitly mocked; the real adapter, files, model and reconstruction run.
"""
from contextlib import ExitStack
import json
from pathlib import Path
import socket
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch": raise
    raise unittest.SkipTest("Optional torch is not installed") from error

from experiments.headless import builtin_env as be, builtin_evidence as e
from experiments.headless.builtin_replay_history import verify_history
from experiments.learner import builtin_rollout as br, contextual as c, contextual_model as m
from experiments.learner.builtin_rollout import (BuiltinCollector, load_builtin_episode,
    audit_persisted_builtin_policy)
from experiments.learner.contextual_rollout import _wire_text
from tests import test_builtin_env as boundaries
from tests.test_contextual_armies import history_at_armies

initial = boundaries.initial


class SyntheticProcess:
    def __init__(self):
        self.done = threading.Event()
        self.returncode = None
        self.connection = None
        self.error = None

    def poll(self): return self.returncode

    def wait(self, timeout=None):
        if not self.done.wait(timeout): raise subprocess.TimeoutExpired("synthetic-transport", timeout)
        return self.returncode

    def end(self, code):
        self.returncode = code
        self.done.set()

    def stop(self):
        self.end(-15)
        if self.connection is not None:
            try: self.connection.shutdown(socket.SHUT_RDWR)
            except OSError: pass


class SyntheticLauncher:
    def __init__(self, root, boundary, profile=c.DIALOGS_PROFILE):
        self.profile = profile
        self.boundary = boundary
        self.binary = root / "bin/vcmiclient"
        self.ai_library = root / "bin/AI/libExplorerRL.dylib"
        self.ai_library.parent.mkdir(parents=True)
        self.binary.write_bytes(b"heroes-contextual-terminal-barrier-v1")
        self.ai_library.write_bytes(" ".join((profile.environment, profile.history_schema, profile.action_schema)).encode())
        for name in ("vcmireplay", "libvcmi.dylib", "AI/libBattleAI.dylib", "AI/libNullkiller.dylib"):
            (self.binary.parent / name).write_bytes(b"EXPLORER_RL_BUILTIN_OPPONENT")
        self.runtime = e.native.runtime_identity(self.binary.parent)
        self.process = SyntheticProcess()
        self.responses = []
        self.environment = None

    def preflight(self): pass

    def launch(self, folder, seed, environment, log):
        self.environment = environment
        self.thread = threading.Thread(target=self.exchange, args=(folder, environment), daemon=True)
        self.thread.start()
        return self.process, {"synthetic": True}

    def exchange(self, folder, env):
        if self.boundary == "startup_failure":
            self.process.end(2)
            return
        player, episode = int(env["EXPLORER_RL_PLAYERS"]), env["EXPLORER_RL_EPISODE"]
        base = {"protocol_version": self.profile.protocol, "episode": episode, "player": player}
        try:
            with socket.create_connection(("127.0.0.1", int(env[f"EXPLORER_RL_PORT_{player}"])), timeout=5) as sock:
                self.process.connection = sock
                stream = sock.makefile("rb")
                def send(message): sock.sendall((json.dumps(message) + "\n").encode())
                def batch(*records): send({**base, "type": "contextual", "records": records})
                observation = c.encode_frame(initial(player, self.profile).frames[0])["observation"]
                def decision(index):
                    return {"kind": "decision", "day": 1, "decision_id": index,
                            "revision": index, "observation": observation}
                send({**base, "type": "ready", "interface": self.profile.environment,
                      "history_schema": self.profile.history_schema, "action_schema": self.profile.action_schema})
                batch(decision(0))
                reply = json.loads(stream.readline())
                self.responses.append(reply)
                if reply["action"]["kind"] != "end_turn": raise AssertionError("Only offered action is end_turn")
                batch({"kind": "consumed", "day": 1, "decision_id": 0, "consumer": "engine"},
                      {"kind": "issued", "day": 1, "decision_id": 0, "command_index": 0},
                      {"kind": "acknowledged", "day": 1, "decision_id": 0, "command_index": 0, "accepted": True},
                      {"kind": "result", "day": 1, "decision_id": 0,
                       "result": {"status": "turn_ended", "accepted": True, "position": None,
                                  "movement": None, "destination_reached": None}})
                if self.boundary == "natural":
                    (folder / "engine").mkdir()
                    for name, record in zip(e.native.FILES, boundaries.BuiltinReceiptTests().records(player)):
                        record["episode"] = episode
                        record["interface"] = self.profile.environment
                        (folder / name).write_text(json.dumps(record) + "\n")
                    batch({"kind": "terminal", "day": 1, "outcome": ("win", "loss")[player]})
                else:
                    batch(decision(1))
                    extra = stream.readline()
                    if extra: raise AssertionError("Cutoff sent an unsolicited action")
                stream.close()
            if self.process.returncode is None: self.process.end(0)
        except BaseException as error:
            self.process.error = error
            self.process.end(2)


class BuiltinCollectorIntegrationTests(unittest.TestCase):
    interface = c.DIALOGS_ENVIRONMENT

    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def run_case(self, root, player, boundary):
        profile = e.profile_for(self.interface)
        launcher = SyntheticLauncher(root, boundary, profile)
        identity = {"engine_revision": "synthetic-not-native", "map_sha256": "b"*64,
                    "native_runtime_sha256": launcher.runtime}
        opponent = e.opponent_identity(player, launcher.runtime["AI/libNullkiller.dylib"])
        torch.manual_seed(172)
        model = m.ContextualActorCritic(m.ModelConfig(schema=m.model_schema_for(profile.history_schema), hidden_size=8, feature_size=4))
        collector = BuiltinCollector(model, "synthetic-frozen-checkpoint", learner_player=player,
                                     opponent_identity=opponent, policy_seed=17, interface=self.interface)
        env = be.BuiltinHeadlessEnv(launcher, root / "episodes", learner_player=player, interface=self.interface)
        with ExitStack() as stack:
            stack.enter_context(patch.object(env, "_identity", return_value=identity))
            stack.enter_context(patch.object(env, "_engine_evidence", return_value=({"synthetic": True}, None)))
            stack.enter_context(patch.object(e, "capture_files", return_value={"schema":e.REPLAY_SCHEMA,"recorded":True,"files":{}}))
            stack.enter_context(patch.object(be.native, "terminal_capture_close"))
            stack.enter_context(patch.object(be, "stop_process_group", side_effect=lambda p:p.stop()))
            try:
                collected = collector.run_episode(env, episode=1, seed=11, max_requests=1)
            finally:
                env.close()
        launcher.thread.join(2)
        self.assertFalse(launcher.thread.is_alive())
        self.assertIsNone(launcher.process.error)
        self.assertEqual(launcher.environment["EXPLORER_RL_PLAYERS"], str(player))
        self.assertEqual(launcher.environment["EXPLORER_RL_BUILTIN_OPPONENT"], "Nullkiller")
        self.assertNotIn(f"EXPLORER_RL_PORT_{1-player}", launcher.environment)
        folder = env.episode_dir
        (folder / "builtin-rollout.json").write_text(json.dumps(collected.to_dict()))
        return collected, folder, model, opponent, launcher, collector

    def test_both_seats_full_socket_boundary_and_persisted_reconstruction(self):
        for player in (0,1):
            for boundary in ("cutoff", "natural"):
                with self.subTest(player=player,boundary=boundary), tempfile.TemporaryDirectory() as tmp:
                    got, folder, model, opponent, launcher, collector = self.run_case(Path(tmp),player,boundary)
                    self.assertEqual(len(got.decisions),1)
                    self.assertEqual(len(launcher.responses),1)
                    self.assertEqual(len(collector.sessions),1)
                    self.assertEqual(set(got.manifest["players"]),{e.COLORS[player]})
                    self.assertFalse(got.optimization_eligible)
                    self.assertEqual(got.training_samples,0)
                    self.assertFalse(any("opponent_checkpoint" in x for x in got.to_dict()["decisions"]))
                    self.assertEqual(load_builtin_episode(folder/"builtin-rollout.json"),got)
                    audit = audit_persisted_builtin_policy(folder,model,got.checkpoint,17,opponent)
                    self.assertTrue(audit["passed"])
                    self.assertEqual(audit["actions_resampled"],1)
                    self.assertEqual(audit["values_recomputed"],2 if boundary=="cutoff" else 1)
                    self.assertFalse(audit["history_audit"]["native_execution_verified"])
                    final = e.read_finalization(folder,got.manifest)
                    reconstructed = verify_history(folder,got.manifest,final)
                    self.assertTrue(reconstructed["verified"])
                    if boundary=="cutoff":
                        self.assertEqual(got.manifest["outcomes_by_player"],dict.fromkeys(e.COLORS))
                    else:
                        self.assertEqual(got.manifest["rewards_by_player"],{"red":1,"blue":-1})

    def test_empty_startup_retains_explicit_unscored_incomplete_failure(self):
        for player in (0,1):
            with self.subTest(player=player), tempfile.TemporaryDirectory() as tmp:
                got,folder,model,opponent,launcher,_=self.run_case(Path(tmp),player,"startup_failure")
                self.assertEqual(got.manifest["status"],"failed")
                self.assertEqual(got.manifest["outcomes_by_player"],dict.fromkeys(e.COLORS))
                self.assertFalse(got.manifest["players"][e.COLORS[player]][e.profile_for(self.interface).history_key]["complete"])
                self.assertEqual([f.kind for f in got.history],["failure"])
                self.assertEqual(got.decisions,())
                self.assertEqual(got.estimates,())
                self.assertEqual(launcher.responses,[])
                self.assertTrue((folder/"manifest.json").is_file())
                self.assertTrue(audit_persisted_builtin_policy(folder,model,got.checkpoint,17,opponent)["passed"])

    def test_artifact_and_disk_tampering_are_independently_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            got,folder,model,opponent,_,_=self.run_case(Path(tmp),0,"cutoff")
            for mutate in (lambda r:r.update(player=1),
                           lambda r:r.update(optimization_eligible=True),
                           lambda r:r["decisions"][0].update(opponent_checkpoint="fabricated"),
                           lambda r:r["opponent_identity"].update(player=True),
                           lambda r:r["estimates"][0]["prefix"].update(history_sha256="f"*64),
                           lambda r:r["decisions"].clear()):
                raw=got.to_dict();mutate(raw)
                with self.assertRaises(ValueError):load_builtin_episode(raw)
            path=folder/got.manifest["players"]["red"][e.profile_for(self.interface).history_key]["path"]
            path.write_bytes(path.read_bytes()+b"\n")
            with self.assertRaises(ValueError):audit_persisted_builtin_policy(folder,model,got.checkpoint,17,opponent)

    def test_environment_frames_are_checked_against_records_and_change_nothing(self):
        handed = []

        def checked(records, typed):
            handed.append((records, typed))
            return delivered(records, typed)
        delivered = br._delivered
        for player in (0, 1):
            for boundary in ("cutoff", "natural"):
                with self.subTest(player=player, boundary=boundary), tempfile.TemporaryDirectory() as tmp:
                    handed.clear()
                    with patch.object(br, "_delivered", checked):
                        typed = self.run_case(Path(tmp) / "typed", player, boundary)[0]
                    # The request, the cutoff decision delivered again at truncation, then the final frames.
                    self.assertEqual(len(handed), 3 if boundary == "cutoff" else 2)
                    for records, frames in handed:
                        self.assertIs(type(frames), tuple)
                        self.assertEqual([c.canonical_frame(frame) for frame in frames], list(map(_wire_text, records)))
                    with patch.object(br, "_delivered", lambda records, typed: records):
                        decoded = self.run_case(Path(tmp) / "decoded", player, boundary)[0]
                    typed, decoded = typed.to_dict(), decoded.to_dict()
                    for run in (typed, decoded):  # wall-clock fields, and the proof digest covering timestamped logs
                        del run["manifest"]["created_at"], run["manifest"]["wall_seconds"], run["manifest"]["evidence_ref"]
                    self.assertEqual(typed, decoded)

    def test_feature_presence_checks_bound_rules_library_not_client(self):
        with tempfile.TemporaryDirectory() as tmp:
            launcher=SyntheticLauncher(Path(tmp),"cutoff")
            env=be.BuiltinHeadlessEnv(launcher,Path(tmp)/"out",learner_player=0)
            self.assertNotIn(b"EXPLORER_RL_BUILTIN_OPPONENT",launcher.binary.read_bytes())
            env._preflight_contextual()
            (launcher.binary.parent/"libvcmi.dylib").write_bytes(b"older-rules")
            with self.assertRaisesRegex(ValueError,"rules library"):env._preflight_contextual()


class DeliveredFramesTests(unittest.TestCase):
    def frames(self):
        frames = tuple(history_at_armies().frames)
        return frames, [json.loads(c.canonical_frame(frame)) for frame in frames]

    def test_exact_frames_replace_their_records(self):
        frames, records = self.frames()
        self.assertIs(br._delivered(records, None), records)
        self.assertIs(br._delivered(records, frames), frames)
        self.assertEqual(tuple(map(c.decode_frame, records)), frames)

    def test_every_difference_is_rejected(self):
        frames, records = self.frames()
        other = tuple(history_at_armies(player=1).frames)
        changed = json.loads(json.dumps(records))
        changed[-1]["revision"] += 1
        unencodable = [*records[:-1], {**records[-1], "revision": float("nan")}]
        for typed, delivered in ((frames[:-1], records), (frames, records[:-1]), ((), records),
                                 (frames[::-1], records), (frames, records[::-1]), (other, records),
                                 (frames, changed), (frames, unencodable), (list(frames), records),
                                 (frames, tuple(records)), ((*frames[:-1], records[-1]), records)):
            with self.subTest(typed=len(typed), delivered=len(delivered)), self.assertRaises(ValueError):
                br._delivered(delivered, typed)


if __name__ == "__main__": unittest.main()
