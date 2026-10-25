"""Real mutex regression for native request evidence, without a game/runtime."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tests.test_builtin_native_contract import function

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "vcmi"


@unittest.skipUnless((ENGINE / "client/Client.cpp").is_file(), "Prepared pinned source required")
class NativeRequestLockOrderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("c++")
        if compiler is None:
            raise unittest.SkipTest("C++20 compiler required")
        cls.temporary = tempfile.TemporaryDirectory(prefix="heroes-request-lock-")
        cls.directory = Path(cls.temporary.name)
        native = function((ENGINE / "client/Client.cpp").read_text(),
            "int CClient::sendRequest(const CPackForServer & request, PlayerColor player, bool waitTillRealize)")
        fixture = (ROOT / "tests/native/native_request_lock_order.cpp").read_text()
        assert fixture.count("// PRODUCTION_SEND_REQUEST") == 1
        source = cls.directory / "test.cpp"
        source.write_text(fixture.replace("// PRODUCTION_SEND_REQUEST", native))
        cls.binary = cls.directory / "test"
        result = subprocess.run([compiler, "-std=c++20", "-pthread", "-Wall", "-Wextra", "-I", str(ROOT),
            str(source), "-o", str(cls.binary)], capture_output=True, text=True, timeout=30)
        if result.returncode:
            raise RuntimeError(result.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def run_case(self, mode):
        result = subprocess.run([str(self.binary), mode], capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, mode + " passed\n")

    def test_native_wait_releases_inversion_restores_before_capture_and_waits_for_ack(self):
        self.run_case("contended")

    def test_unselected_and_nonblocking_paths_preserve_lock_behavior(self):
        for mode in ("legacy", "learner", "no-wait", "nested-no-wait"):
            with self.subTest(mode=mode):
                self.run_case(mode)

    def test_constructor_and_capture_failures_restore_shared_ownership_and_release_event(self):
        for mode in ("event-failure", "capture-failure"):
            with self.subTest(mode=mode):
                self.run_case(mode)
