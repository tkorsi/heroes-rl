"""Parallel verification: forked checks return exactly what an in-process call would, and keep the serial error order."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from experiments.headless import background_check, builtin_replay as replay
from experiments.headless.background_check import BackgroundCheck
from tests import test_builtin_replay_diagnostic as diagnostic


def running(pid):
    try:
        with open(f"/proc/{pid}/stat") as stream:
            return stream.read().split(")")[-1].split()[0] not in ("Z", "X")
    except FileNotFoundError:
        return False


def reaped(pid):
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        return True
    return False


def answer(value):
    return {"value": value, "pid": os.getpid()}


def hang():
    time.sleep(60)


def sleeper(path):
    # As in the native replay: subprocess.run kills its process when an exception unwinds it.
    class Recording(subprocess.Popen):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            Path(path).write_text(str(self.pid))
    subprocess.Popen = Recording  # this forked child only
    subprocess.run(["sleep", "60"])


def descriptors():
    return sorted(os.listdir("/proc/self/fd"))


@unittest.skipUnless(hasattr(os, "fork"), "Background checks fork")
class BackgroundCheckTests(unittest.TestCase):
    def test_value_and_exception_type_are_returned_exactly(self):
        check = BackgroundCheck(answer, (1, 2.5, None))
        result = check.result(timeout=30)
        self.assertEqual(result["value"], (1, 2.5, None))  # tuples stay tuples
        self.assertNotEqual(result["pid"], os.getpid())
        self.assertTrue(reaped(check.pid))
        with self.assertRaisesRegex(ValueError, "invalid literal"):
            BackgroundCheck(int, "not a number").result(timeout=30)

    def test_patches_made_before_the_fork_apply_in_the_child(self):
        with patch.object(background_check, "_CHUNK", 7):
            self.assertEqual(BackgroundCheck(lambda: background_check._CHUNK).result(timeout=30), 7)

    def test_timeout_and_stop_end_and_reap_the_child(self):
        check = BackgroundCheck(hang)
        with self.assertRaises(TimeoutError):
            check.result(timeout=0.2)
        self.assertTrue(reaped(check.pid))
        other = BackgroundCheck(hang)
        other.stop()
        self.assertTrue(reaped(other.pid))
        other.stop()  # idempotent

    @unittest.skipUnless(sys.platform.startswith("linux"), "Process states use /proc")
    def test_stop_unwinds_the_childs_own_subprocess(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "pid"
            check = BackgroundCheck(sleeper, path)
            deadline = time.monotonic() + 10
            while not path.exists() or not path.read_text():
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.01)
            grandchild = int(path.read_text())
            check.stop()
            deadline = time.monotonic() + 5
            while running(grandchild) and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertFalse(running(grandchild))

    @unittest.skipUnless(sys.platform.startswith("linux"), "Descriptor listing uses /proc")
    def test_the_child_keeps_only_its_result_pipe_and_standard_streams(self):
        read, write = os.pipe()
        self.addCleanup(os.close, read)
        self.addCleanup(os.close, write)
        self.assertGreater(len(descriptors()), 5)  # the caller holds more, including this pipe
        names = BackgroundCheck(descriptors).result(timeout=30)
        self.assertLessEqual(len(names), 5)  # 0, 1, 2, the result pipe, and the listing's own directory


class OverlappedNativeReplayTests(unittest.TestCase):
    """verify_episode runs its Python history check while it waits on the native verifier."""

    def setUp(self):
        self.fixture = diagnostic.DiagnosticAdmissionTests("test_success_is_supplemental_only_and_preserves_original_files")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def verify(self, *, history, native):
        f = self.fixture
        with patch.object(replay, "bound_evidence", return_value=f.final), \
             patch.object(replay, "effective_environment", return_value={"verified": True}), \
             patch.object(replay, "cross_check_wire", return_value=f.wire), \
             patch.object(replay.legacy, "replay_profile", return_value=({}, f.root / "map.vmap")), \
             patch("experiments.headless.builtin_replay_history.verify_history", side_effect=history), \
             patch.object(replay, "run_native", side_effect=native):
            return replay.verify_episode(f.episode, binary=f.root / "vcmireplay", data=f.root, output=f.output)

    def test_both_run_at_once_and_both_results_are_recorded(self):
        started = threading.Event()
        def history(*args):
            started.set()
            return {"verified": True}
        def native(*args, **kwargs):
            self.assertTrue(started.wait(10))  # the history check began before the native verifier finished
            return self.fixture.native
        result = self.verify(history=history, native=native)
        self.assertTrue(result["verified"], result.get("error"))
        self.assertEqual(result["history"], {"verified": True})
        self.assertEqual(result["native"], self.fixture.native)

    def test_a_history_error_is_the_reported_error_whatever_the_native_outcome(self):
        def history(*args):
            raise ValueError("history differs")
        for native in (lambda *a, **k: self.fixture.native, ValueError("native differs")):
            with self.subTest(native=native):
                self.fixture.doCleanups()
                self.fixture.setUp()
                result = self.verify(history=history, native=native)
                self.assertFalse(result["verified"])
                self.assertIn("history differs", result["error"])
                self.assertNotIn("native", result)

    def test_a_native_error_after_a_passing_history_keeps_the_history(self):
        result = self.verify(history=lambda *args: {"verified": True}, native=subprocess.TimeoutExpired("vcmireplay", 1))
        self.assertFalse(result["verified"])
        self.assertEqual(result["history"], {"verified": True})
        self.assertIn("TimeoutExpired", result["error"])
        self.assertNotIn("native", result)


if __name__ == "__main__":
    unittest.main()
