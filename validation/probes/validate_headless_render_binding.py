#!/usr/bin/env python3
"""Validate renderer evidence binding with synthetic fixtures and real FFmpeg."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probe(name, fixture_type, renderer, replay):
    fixture = fixture_type()
    fixture.setUp()
    original_renderer = renderer.TraceRenderer
    try:
        episode = fixture.episode(verified=True)
        manifest = episode / "manifest.json"
        trace = episode / "verification/trace.jsonl"
        report = episode / "verification/replay-verification.json"
        output = fixture.base / "render.mp4"
        original = {"manifest_sha256": sha(manifest), "trace_sha256": sha(trace)}
        original_report = sha(report)

        def mutate_trace():
            rows = [json.loads(line) for line in trace.read_text().splitlines()]
            rows[-1]["objects"][0][1] = 19
            rows[-1]["players"]["0"]["visible"] = "1" * 400
            trace.write_text("".join(json.dumps(row) + "\n" for row in rows))

        def mutate_manifest():
            data = json.loads(manifest.read_text())
            data["outcomes_by_player"] = {"red": "loss", "blue": "win"}
            manifest.write_text(json.dumps(data))

        if name in ("trace", "both"):
            mutate_trace()
        if name in ("manifest", "both"):
            mutate_manifest()
        if name in ("trace_whitespace", "manifest_whitespace"):
            path = trace if name.startswith("trace") else manifest
            path.write_bytes(path.read_bytes() + b"\n")
        if name == "unbound_report":
            report.write_text(json.dumps({"verified": True}))
        if name == "failed_report":
            data = json.loads(report.read_text())
            data["verified"] = False
            report.write_text(json.dumps(data))
        if name == "failed_reverification":
            # The synthetic manifest lacks replay evidence; failure occurs before any process launch.
            result = replay.verify_episode(episode, binary=Path(sys.executable), data=fixture.base, trace=True)
            saved = json.loads(report.read_text())
            return {"case": name, "passed": result["verified"] is False and saved["verified"] is False
                    and not trace.exists(), "old_report_replaced": sha(report) != original_report,
                    "stale_trace_removed": not trace.exists(), "verified": result["verified"], "error": result.get("error")}
        if name == "mutate_after_input_read":
            class MutatingRenderer(original_renderer):
                def __init__(self, header, data):
                    super().__init__(header, data)
                    mutate_trace()
                    mutate_manifest()
            renderer.TraceRenderer = MutatingRenderer

        expected_accept = name in ("valid", "mutate_after_input_read")
        try:
            result = renderer.render(episode, output, fps=5, max_seconds=3, hold_seconds=0)
        except renderer.video.VideoError as error:
            return {"case": name, "expected_accept": expected_accept, "accepted": False,
                    "passed": not expected_accept and not output.exists(), "error": str(error),
                    "output_absent": not output.exists(), "verification_report_unchanged": sha(report) == original_report}
        hashes_match = all(result.get(key) == value for key, value in original.items())
        return {"case": name, "expected_accept": expected_accept, "accepted": True,
                "passed": expected_accept and result["replay_verified"] is True and hashes_match,
                "replay_verified": result["replay_verified"], "outcomes": result["outcomes"],
                "recorded_hashes_match_original_bytes": hashes_match,
                "current_files_differ": sha(trace) != original["trace_sha256"] or sha(manifest) != original["manifest_sha256"],
                "video_sha256": sha(output), "codec": result["codec"], "frames_written": result["frames_written"]}
    finally:
        renderer.TraceRenderer = original_renderer
        fixture.tearDown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source, output = args.source.expanduser().resolve(), args.output.expanduser().resolve()
    sys.path.insert(0, str(source / "tests"))
    from test_render_replay import RenderReplayTests, render_replay
    from test_replay import replay

    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("Requires actual FFmpeg and ffprobe")
    cases = [probe(name, RenderReplayTests, render_replay, replay) for name in
             ("valid", "trace", "manifest", "both", "trace_whitespace", "manifest_whitespace",
              "unbound_report", "failed_report", "failed_reverification", "mutate_after_input_read")]
    videos_identical = cases[0]["video_sha256"] == cases[-1]["video_sha256"]
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*replay.py"]
    tests = subprocess.run(command, cwd=source, text=True, capture_output=True, check=False)
    result = {
        "schema_version": 1, "kind": "renderer-binding-fix-validation",
        "reviewed_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip(),
        "source_checkout": str(source), "cases": cases,
        "same_video_after_files_change_during_render": videos_identical,
        "passed": all(case["passed"] for case in cases) and videos_identical and tests.returncode == 0,
        "tests": {"command": command, "returncode": tests.returncode, "output": tests.stdout + tests.stderr},
        "ffmpeg": ffmpeg, "ffprobe": ffprobe,
        "ffmpeg_version": subprocess.check_output([ffmpeg, "-version"], text=True).splitlines()[0],
        "source_sha256": {name: sha(source / name) for name in
                          ("experiments/headless/replay.py", "experiments/headless/render_replay.py")},
        "probe_sha256": sha(Path(__file__)),
        "scope": "Synthetic fixture with correctly bound report and real FFmpeg encoding; no real game, native replay or training.",
        "stale_probe": "An existing successful report and trace are removed before a reverification fails for absent replay evidence.",
        "race_probe": "Temporary TraceRenderer subclass changes disk files after renderer reads and hashes them. Video and provenance still use the original captured bytes.",
        "limits": "Trusts the native replay writer and local verification report; this is input integrity, not cryptographic authenticity against a writer replacing both content and hashes.",
        "temporary_artifacts": "All fixture inputs and videos removed; only this probe and compact report remain.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"reviewed_commit": result["reviewed_commit"], "passed": result["passed"],
                      "cases": {case["case"]: case["passed"] for case in cases}, "report": str(output)}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
