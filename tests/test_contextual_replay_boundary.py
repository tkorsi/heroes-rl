"""Synthetic rejection checks; these do not establish native replay fidelity."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments/headless" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


replay, renderer = load("replay"), load("render_replay")


class ContextualReplayBoundaryTests(unittest.TestCase):
    def test_each_explicit_contextual_marker_is_rejected(self):
        for key, value in {
            "kind": "contextual-headless-self-play", "environment": "adventure-windows-v1",
            "interface": "adventure-windows-v1", "history_schema": "heroes-player-history-v2",
            "action_schema": "heroes-context-actions-v2", "protocol_version": 4, "type": "contextual",
        }.items():
            with self.subTest(key=key), self.assertRaisesRegex(replay.ReplayError, "unsupported"):
                replay.require_legacy_interface({key: value})

    def test_verification_and_tamper_preserve_existing_output_and_do_not_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            episode = Path(directory)
            (episode / "manifest.json").write_text(json.dumps({"environment": "adventure-windows-v1"}))
            output = episode / "existing"
            output.mkdir()
            sentinel = output / "preserve.txt"
            sentinel.write_text("existing evidence")
            for function in (replay.verify_episode, replay.tamper_episode):
                with self.subTest(function=function.__name__), mock.patch.object(replay, "run_replay") as run:
                    with self.assertRaisesRegex(replay.ReplayError, "unsupported"):
                        function(episode, binary=episode / "missing-binary", data=episode, output=output)
                    run.assert_not_called()
                    self.assertEqual(sentinel.read_text(), "existing evidence")

    def test_contextual_wire_cannot_pass_an_empty_legacy_capture_check(self):
        with tempfile.TemporaryDirectory() as directory:
            bridge, observations = Path(directory) / "bridge.jsonl", Path(directory) / "observations.jsonl"
            bridge.write_text(json.dumps({"direction": "engine", "message": {
                "type": "contextual", "protocol_version": 4, "records": []}}) + "\n")
            observations.write_text("")
            with self.assertRaisesRegex(replay.ReplayError, "unsupported"):
                replay.cross_check_player(bridge, observations, cut_short=False)

    def test_renderer_rejects_rebound_hashes_and_forged_legacy_verified_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            episode = Path(directory)
            verification = episode / "verification"
            verification.mkdir()
            manifest = json.dumps({"kind": "contextual-headless-self-play"}).encode()
            trace = b"{}\n"
            (episode / "manifest.json").write_bytes(manifest)
            (verification / "trace.jsonl").write_bytes(trace)
            (verification / "replay-verification.json").write_text(json.dumps({
                "verified": True, "schema_version": 2,
                "inputs": {"manifest_sha256": hashlib.sha256(manifest).hexdigest(),
                           "trace_sha256": hashlib.sha256(trace).hexdigest()},
            }))
            with mock.patch.object(renderer.subprocess, "Popen") as launch:
                with self.assertRaisesRegex(renderer.video.VideoError, "unsupported"):
                    renderer.render(episode, episode / "output.mp4")
                launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
