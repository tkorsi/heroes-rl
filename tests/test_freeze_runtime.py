"""Freezing a build into a runtime folder, with the macOS tools replaced by a recorder."""
from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import freeze_runtime  # noqa: E402

FILES = (*freeze_runtime.PROGRAMS, *(f"AI/lib{plugin}.dylib" for plugin in freeze_runtime.PLUGINS))


@pytest.fixture
def build(tmp_path):
    folder = tmp_path / "vcmi" / "build" / "bin"
    for name in FILES:
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_text(name)
    for name in ("config", "Mods"):
        (folder / name / "inner").mkdir(parents=True)
        (folder / name / "inner" / "file.json").write_text("{}")
    return folder


@pytest.fixture
def commands(build, monkeypatch):
    """What the freeze asked of ``otool``, ``install_name_tool`` and ``codesign``.

    ``otool`` names the build folder in a copy until ``install_name_tool`` has changed that copy.
    """
    relinked, recorded = set(), []

    def run(*command):
        recorded.append(command)
        if command[0] == "install_name_tool":
            relinked.add(command[-1])
        return f"path {build} (offset 12)" if command[0] == "otool" and command[-1] not in relinked else ""

    monkeypatch.setattr(freeze_runtime, "run", run)
    return recorded


def files(folder):
    return sorted(str(path.relative_to(folder)) for path in folder.rglob("*") if path.is_file())


def test_a_runtime_appears_whole_and_only_once_it_is_ready(build, commands, tmp_path, monkeypatch):
    runtime = tmp_path / "runtime" / "new"
    there, real = [], shutil.copytree

    def copytree(*arguments, **options):
        there.append(runtime.exists())  # config and Mods are copied last, folder by folder
        return real(*arguments, **options)

    monkeypatch.setattr(shutil, "copytree", copytree)
    client = freeze_runtime.freeze(build, runtime)
    assert there == [False] * 4  # nothing stood at the final place while the runtime was incomplete
    assert client == runtime / "bin" / "vcmiclient" and client.read_text() == "vcmiclient"
    assert files(runtime / "bin") == sorted([*FILES, "config/inner/file.json", "Mods/inner/file.json"])
    assert [item.name for item in runtime.parent.iterdir()] == ["new"]  # the place it was put together is gone
    # Every copy was pointed beside itself and signed again; a plugin looks one folder up.
    relinked = {Path(command[-1]).name: command[-2] for command in commands if command[0] == "install_name_tool"}
    assert len(relinked) == len(FILES)
    assert relinked["vcmiclient"] == "@loader_path" and relinked["libExplorerRL.dylib"] == "@loader_path/.."
    assert sum(command[0] == "codesign" for command in commands) == len(FILES)


def test_a_failed_freeze_leaves_nothing_and_the_same_command_then_works(build, commands, tmp_path):
    runtime = tmp_path / "runtime" / "new"
    (build / "vcmiserver").unlink()  # the second program is missing from the build
    with pytest.raises(FileNotFoundError):
        freeze_runtime.freeze(build, runtime)
    assert list(runtime.parent.iterdir()) == []  # neither a runtime with only vcmiclient in it nor a leftover
    (build / "vcmiserver").write_text("vcmiserver")  # the build is repaired
    assert freeze_runtime.freeze(build, runtime).is_file()
    assert [item.name for item in runtime.parent.iterdir()] == ["new"]


def test_a_copy_that_still_names_the_build_folder_is_not_published(build, tmp_path, monkeypatch):
    runtime = tmp_path / "runtime" / "new"
    monkeypatch.setattr(freeze_runtime, "run", lambda *command: str(build) if command[0] == "otool" else "")
    with pytest.raises(SystemExit, match="vcmiclient still refers to the build folder"):
        freeze_runtime.freeze(build, runtime)
    assert list(runtime.parent.iterdir()) == []


def test_an_existing_runtime_is_refused_and_left_as_it_is(build, commands, tmp_path):
    runtime = tmp_path / "runtime" / "old"
    (runtime / "bin").mkdir(parents=True)
    (runtime / "bin" / "vcmiclient").write_text("the engine training uses")
    with pytest.raises(SystemExit, match="exists already"):
        freeze_runtime.freeze(build, runtime)
    assert files(runtime) == ["bin/vcmiclient"] and commands == []
    assert (runtime / "bin" / "vcmiclient").read_text() == "the engine training uses"
    assert [item.name for item in runtime.parent.iterdir()] == ["old"]
