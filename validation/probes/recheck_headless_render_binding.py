#!/usr/bin/env python3
"""Synthetic renderer evidence-binding probe; no real game or engine replay.

Exit 0 confirms the known defect was reproduced and existing renderer tests
passed. It is not a passing acceptance test for corrected render behavior.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probe(name: str, alter_trace: bool, alter_manifest: bool, fixture_type, renderer) -> dict:
    fixture = fixture_type()
    fixture.setUp()
    try:
        episode = fixture.episode(verified=True)
        trace = episode / "verification/trace.jsonl"
        manifest_path = episode / "manifest.json"
        verification = episode / "verification/replay-verification.json"
        original = {"trace": digest(trace), "manifest": digest(manifest_path), "verification": digest(verification)}
        if alter_trace:
            rows = [json.loads(line) for line in trace.read_text().splitlines()]
            rows[-1]["objects"][0][1] = 19
            rows[-1]["players"]["0"]["visible"] = "1" * 400
            trace.write_text("".join(json.dumps(row) + "\n" for row in rows))
        if alter_manifest:
            manifest = json.loads(manifest_path.read_text())
            manifest["outcomes_by_player"] = {"red": "loss", "blue": "win"}
            manifest_path.write_text(json.dumps(manifest))
        modified = {"trace": digest(trace), "manifest": digest(manifest_path), "verification": digest(verification)}
        video = fixture.base / "probe.mp4"
        result = renderer.render(episode, video, fps=5, max_seconds=3, hold_seconds=0)
        return {
            "case": name,
            "alter_trace": alter_trace,
            "alter_manifest": alter_manifest,
            "original_sha256": original,
            "rendered_input_sha256": modified,
            "verification_report_unchanged": original["verification"] == modified["verification"],
            "accepted": True,
            "replay_verified": result["replay_verified"],
            "outcomes": result["outcomes"],
            "codec": result["codec"],
            "frames_written": result["frames_written"],
            "video_sha256": digest(video),
        }
    except Exception as error:
        return {"case": name, "accepted": False, "error": f"{type(error).__name__}: {error}"}
    finally:
        fixture.tearDown()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="PR source checkout containing tests/test_render_replay.py")
    parser.add_argument("--output", type=Path, required=True, help="Result JSON path")
    args = parser.parse_args()
    source, output = args.source.expanduser().resolve(), args.output.expanduser().resolve()
    if not (source / "tests/test_render_replay.py").is_file():
        parser.error("--source must contain tests/test_render_replay.py")
    sys.path.insert(0, str(source / "tests"))
    from test_render_replay import RenderReplayTests, render_replay

    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("This probe requires real ffmpeg and ffprobe binaries")
    cases = [probe(name, trace, manifest, RenderReplayTests, render_replay) for name, trace, manifest in
             [("baseline", False, False), ("changed_trace", True, False),
              ("changed_manifest", False, True), ("changed_trace_and_manifest", True, True)]]
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_render_replay.py"]
    tests = subprocess.run(command, cwd=source, capture_output=True, text=True, check=False)
    result = {
        "schema_version": 1,
        "kind": "renderer-input-binding-recheck",
        "reviewed_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip(),
        "source_checkout": str(source),
        "scope": "Synthetic fixture and actual FFmpeg encoding; no real-game launch, engine replay, training or source edits.",
        "fixture": "tests/test_render_replay.py RenderReplayTests.episode(verified=True)",
        "fixture_limit": "The verification report is synthetic. This checks renderer input binding, not native replay fidelity.",
        "trace_change": "Final-frame first hero x becomes 19; red visible tiles become 400 ones.",
        "manifest_change": "Swap the synthetic original red-win/blue-loss outcomes to red-loss/blue-win.",
        "temporary_artifacts": "All fixture inputs and encoded videos removed after each case.",
        "ffmpeg": ffmpeg,
        "ffprobe": ffprobe,
        "ffmpeg_version": subprocess.check_output([ffmpeg, "-version"], text=True).splitlines()[0],
        "cases": cases,
        "finding_reproduced": all(case.get("accepted") and case.get("replay_verified") for case in cases),
        "tests": {"command": command, "returncode": tests.returncode, "output": tests.stdout + tests.stderr},
        "source_sha256": {name: digest(source / name) for name in ["experiments/headless/render_replay.py", "experiments/headless/replay.py"]},
        "probe_sha256": digest(Path(__file__)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"reviewed_commit": result["reviewed_commit"], "finding_reproduced": result["finding_reproduced"],
                      "cases": {case["case"]: case.get("accepted") for case in cases}, "tests_returncode": tests.returncode,
                      "report": str(output)}))
    return 0 if result["finding_reproduced"] and tests.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
