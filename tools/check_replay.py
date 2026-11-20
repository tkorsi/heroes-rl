#!/usr/bin/env python3
"""Play one short real game with its replay ledger and have ``vcmireplay`` verify it.

    python tools/check_replay.py STATE [--vcmi CLIENT] [--replay VCMIREPLAY] [--max-days 3] [--seed 1]

STATE is a state folder of its own with a checkpoint and ``latest.json``. This
starts the real engine, so by rule 1 of AGENTS.md the user launches it. Nothing
is trained or saved; the game is played and replayed in a folder of its own
inside STATE, which is removed when every check passed and kept, with the
verifier's reports, when one did not.

The engine sends a decision without a second copy of the observation when it
is told ``EXPLORER_RL_DECISION_OBSERVATION=0``, as training tells it; an engine
built before that switch sends the copy. ``vcmireplay`` reads which of the two
forms a capture has from the capture itself. Three replays show that it does,
and that it still compares every record:

1. the capture verifies from an environment without the variable;
2. it verifies with the variable set to the other form;
3. with one recorded answer of the policy replaced by another, it does not.

One JSON line reports them; the exit status is 0 only if all three held. Run
it once with the engine training uses and once with ``--vcmi`` naming an
engine from before the switch, both against the ``vcmireplay`` under test.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# What the launcher sets for a game. A replay starts without any of it, like a fresh shell.
OWNED_VARIABLES = ("EXPLORER_RL_", "VCMI_RECORD_", "VCMI_EXPERIMENT_", "VCMI_USER_DATA_DIR", "XDG_")
SWITCH = "EXPLORER_RL_DECISION_OBSERVATION"


def play(config, seed, directory):
    """One seeded game with both seats on the newest weights, its engine folder and ledger kept."""
    import torch
    from heroes import actor, checkpoint
    from heroes.engine.env import Env
    path = checkpoint.latest(config.state)
    if path is None:
        raise SystemExit(f"{config.state} has no latest.json")
    torch.set_num_threads(1)
    rng = random.Random(seed)
    job = actor.Job(700000 + seed, rng.randrange(1, 2**31 - 1), seed % 2, (rng.getrandbits(62), rng.getrandbits(62)))
    env = Env(config.engine(), config.limits)
    try:
        return actor.play(env, checkpoint.load_model(path), job, directory, guard=config.guard)
    finally:
        env.close()


def whole_lines(path):
    """The complete lines of a log. An engine stopped at a limit may leave a last line unfinished."""
    data = Path(path).read_bytes()
    return data[:data.rfind(b"\n") + 1].splitlines()


def server_log(lines):
    """The server's log up to its last complete step.

    After ``init`` and ``start`` the recorder writes an input before the server
    handles it and ``applied`` after, so a stopped engine leaves at most one
    input without its pair, and ``vcmireplay`` wants pairs.
    """
    kinds = [json.loads(line).get("kind") for line in lines]
    end = 2
    while kinds[end:end + 2] == ["input", "applied"]:
        end += 2
    if kinds[:2] != ["init", "start"] or kinds[end:] not in ([], ["input"]):
        raise SystemExit("The engine's replay.jsonl is not init, start and input/applied pairs")
    return lines[:end]


def repeats_observation(lines):
    """Whether the capture's decisions carry the observation, read from its first decision; None without one."""
    for line in lines:
        if b'"decision"' in line:
            row = json.loads(line)
            if row.get("kind") == "message":
                for record in row["message"].get("records", ()):
                    if record.get("kind") == "decision":
                        return "observation" in record
    return None


def change_an_answer(lines):
    """The ledger with one answer of the policy replaced by a different one the same seat gave.

    Returns the lines and the number of the changed record, or None when no
    seat gave two different answers.
    """
    answers = [(index, json.loads(line)) for index, line in enumerate(lines) if b'"response"' in line]
    answers = [(index, row) for index, row in answers if row.get("kind") == "response"]
    for index, row in answers[len(answers) // 2:] + answers[:len(answers) // 2]:  # from the middle of the game on
        other = next((item["response"]["action"] for _, item in answers if item["player"] == row["player"]
                      and item["response"]["action"] != row["response"]["action"]), None)
        if other is not None:
            row["response"]["action"] = other
            changed = list(lines)
            changed[index] = json.dumps(row, separators=(",", ":")).encode()
            return changed, row["seq"]
    return None


def replay(binary, profile, folder, name, log, tape, *, prefix, switch=None, timeout=1800):
    """Run ``vcmireplay`` on a capture and return its report, with its exit status."""
    folder.mkdir(parents=True, exist_ok=True)
    files = {"log": folder / f"{name}-replay.jsonl", "contextual": folder / f"{name}-contextual-replay.jsonl"}
    for key, lines in (("log", log), ("contextual", tape)):
        files[key].write_bytes(b"".join(line + b"\n" for line in lines))
    report = folder / f"{name}-report.json"
    command = [str(binary), "--log", str(files["log"]), "--contextual", str(files["contextual"]),
               "--report", str(report)] + (["--prefix"] if prefix else [])
    if sys.platform == "darwin":
        environment = {"VCMI_USER_DATA_DIR": str(profile)}
    else:
        environment = {"XDG_DATA_HOME": str(profile / "data"), "XDG_CONFIG_HOME": str(profile / "config"),
                       "XDG_CACHE_HOME": str(profile / "cache")}
    if switch is not None:
        environment[SWITCH] = switch
    inherited = {key: value for key, value in os.environ.items() if not key.startswith(OWNED_VARIABLES)}
    with (folder / f"{name}-console.log").open("wb") as console:
        result = subprocess.run(command, cwd=Path(binary).parent, env={**inherited, **environment},
                                stdin=subprocess.DEVNULL, stdout=console, stderr=subprocess.STDOUT, timeout=timeout)
    try:
        outcome = json.loads(report.read_text())
    except (OSError, ValueError):
        outcome = {"verified": False, "error": f"vcmireplay wrote no report; see {folder / (name + '-console.log')}"}
    files["contextual"].unlink()  # the largest file by far; the report and the console log stay
    return {**outcome, "returncode": result.returncode}


def check(game_folder, binary, work, *, prefix):
    """The three replays of one kept game; returns the report and whether every check held."""
    engine = Path(game_folder) / "engine"
    if not (engine / "contextual-replay.jsonl").is_file():
        raise SystemExit(f"{engine} has no contextual-replay.jsonl: this engine wrote no replay ledger")
    tape, log = whole_lines(engine / "contextual-replay.jsonl"), server_log(whole_lines(engine / "replay.jsonl"))
    repeated = repeats_observation(tape)
    changed = change_an_answer(tape)
    if repeated is None or changed is None:
        raise SystemExit("The game is too short to check: it needs a decision and two different answers of one seat. "
                         "Raise --max-days.")
    profile = Path(game_folder) / "profile"

    def run(name, lines, switch=None):
        return replay(binary, profile, work, name, log, lines, prefix=prefix, switch=switch)

    def reason(outcome):
        return (outcome.get("error") or json.dumps(outcome.get("divergence")))[:400]

    plain = run("plain", tape)
    other = run("other-environment", tape, switch="0" if repeated else "1")
    tampered = run("changed-answer", changed[0])
    held = {
        "verified": plain.get("verified") is True and plain["returncode"] == 0,
        "form_read_from_capture": plain.get("summary", {}).get("decisions_repeat_observation") is repeated,
        "verified_with_the_other_form_in_the_environment": other.get("verified") is True and other["returncode"] == 0,
        "changed_answer_rejected": tampered.get("verified") is False and tampered["returncode"] != 0,
    }
    report = {"decisions": "repeat the observation" if repeated else "compact", "records": len(tape), **held,
              "checked": plain.get("summary", {}).get("checked"), "changed_record": changed[1]}
    for name, outcome in (("plain", plain), ("other_environment", other)):
        if outcome.get("verified") is not True:
            report[f"{name}_failure"] = reason(outcome)
    report["changed_answer_failure" if held["changed_answer_rejected"] else "changed_answer_report"] = reason(tampered)
    return report, all(held.values())


def main():
    from heroes.config import Config
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("state", type=lambda text: Path(text).expanduser().resolve())
    parser.add_argument("--vcmi", type=lambda text: Path(text).expanduser().resolve(),
                        help="the vcmiclient that plays the game (default: the one training uses)")
    parser.add_argument("--replay", type=lambda text: Path(text).expanduser().resolve(),
                        help="the vcmireplay under test (default: the one beside that vcmiclient)")
    parser.add_argument("--max-days", type=int, default=3, help="stop the game after this game day (default 3)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, help="append the JSON report to this file")
    arguments = parser.parse_args()
    config = Config()
    config = replace(config, state=arguments.state, keep_game_files=True,  # which makes the engine write its ledger
                     limits=replace(config.limits, max_days=arguments.max_days))
    if arguments.vcmi is not None:
        config = replace(config, vcmi=arguments.vcmi)
    binary = arguments.replay or config.vcmi.parent / "vcmireplay"
    config.engine().check()
    if not binary.is_file():
        raise SystemExit(f"vcmireplay not found: {binary}")
    work = Path(tempfile.mkdtemp(prefix="check-replay-", dir=config.state))
    try:
        game = play(config, arguments.seed, work / "game")
        if game.status not in ("completed", "truncated"):
            raise SystemExit(f"The game failed ({game.reason}): {game.detail}")
        report, passed = check(work / "game", binary, work / "replays", prefix=game.status == "truncated")
    except BaseException:
        print(f"The game and what was replayed of it are kept in {work}", file=sys.stderr)
        raise
    report = {"vcmi": str(config.vcmi), "vcmireplay": str(binary), "seed": arguments.seed,
              "game": f"{game.status}:{game.reason}", "day": game.day, "choices": list(game.choices), **report}
    if passed:
        shutil.rmtree(work, ignore_errors=True)
    else:
        report["kept"] = str(work)
    line = json.dumps(report)
    print(line, flush=True)
    if arguments.out is not None:
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        with arguments.out.open("a") as stream:
            stream.write(line + "\n")
    if not passed:
        sys.exit("The replay check failed; the report says which part.")


if __name__ == "__main__":
    main()
