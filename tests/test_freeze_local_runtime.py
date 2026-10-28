"""Path and load-command controls; these tests do not freeze or execute a runtime."""
from pathlib import Path
import tempfile
import unittest

from tools.freeze_local_runtime import ARTIFACTS, expand_token, paths, rpaths


class RuntimeFreezePathTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="heroes-freezer-test-")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "project"
        self.donor = self.root / "vcmi/build/bin"
        self.donor.mkdir(parents=True)
        for name in ARTIFACTS:
            path = self.donor / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b"fixture")
        for name in ("config", "Mods"):
            (self.donor / name).mkdir()
        self.map = self.base / "fixture.vmap"
        self.map.write_bytes(b"fixture")
        self.target = self.base / "durable-runtime"

    def check(self, destination):
        return paths(self.donor, destination, self.map, self.root)

    def test_new_complete_durable_target_is_read_only_preflight(self):
        self.assertEqual(self.check(self.target)[1], self.target.resolve())
        self.assertFalse(self.target.exists())
        (self.donor / "AI/libNullkiller.dylib").unlink()
        with self.assertRaisesRegex(ValueError, "Missing required"):
            self.check(self.target)

    def test_existing_destination_and_dangling_symlink_are_never_reused(self):
        self.target.mkdir()
        with self.assertRaisesRegex(ValueError, "new directory"):
            self.check(self.target)
        self.target.rmdir()
        self.target.symlink_to(self.base / "absent")
        with self.assertRaisesRegex(ValueError, "new directory"):
            self.check(self.target)

    def test_resolved_managed_worktree_and_project_targets_are_rejected(self):
        for path in (self.root / "runtime", self.donor / "runtime"):
            with self.assertRaisesRegex(ValueError, "durable"):
                self.check(path)
        managed = self.base / ".codex/worktrees/check"
        managed.mkdir(parents=True)
        link = self.base / "misleading-durable"
        link.symlink_to(managed)
        with self.assertRaisesRegex(ValueError, "durable"):
            self.check(link / "runtime")

    def test_rpath_parser_and_resolution_preserve_spaces(self):
        text = "cmd LC_RPATH\n cmdsize 88\n path /a build/bin (offset 12)\ncmd LC_RPATH\n cmdsize 32\n path @loader_path/.. (offset 12)\n"
        self.assertEqual(rpaths(text), ["/a build/bin", "@loader_path/.."])
        loader = Path("/durable/bin/AI/libExplorerRL.dylib")
        self.assertEqual(expand_token("@loader_path/../libvcmi.dylib", loader, Path("/durable/bin")),
                         Path("/durable/bin/AI/../libvcmi.dylib"))
        self.assertIsNone(expand_token("@rpath/libvcmi.dylib", loader, Path("/durable/bin")))


if __name__ == "__main__":
    unittest.main()
