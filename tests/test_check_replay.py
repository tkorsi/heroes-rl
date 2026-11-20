"""The replay check on a hand-written capture, with stand-ins for ``vcmireplay``."""
import json
from pathlib import Path
import stat
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import check_replay  # noqa: E402

# A stand-in verifier. It "regenerates" the capture it was first shown and compares, record by record.
# The fixed one reads the form of the decisions from the tape; the other one takes it from its
# environment, as the engine does when nothing tells it otherwise.
VERIFIER = '''#!{python}
import json, os, sys
options = [item for item in sys.argv[1:] if item != "--prefix"]
arguments = dict(zip(options[::2], options[1::2]))
tape = [json.loads(line) for line in open(arguments["--contextual"])]
# Called as for a game that was cut short, with whole steps only and nothing of the launcher's environment.
assert "--prefix" in sys.argv
assert [json.loads(line)["kind"] for line in open(arguments["--log"])] == ["init", "start", "input", "applied"]
assert not any(key.startswith("VCMI_EXPERIMENT_") for key in os.environ)
assert os.environ[{profile!r}].startswith({game!r})
decisions = [record for row in tape if row["kind"] == "message"
             for record in row["message"]["records"] if record["kind"] == "decision"]
repeated = {form}
difference = next((index for index, (seen, made) in enumerate(zip(tape, json.load(open({original!r}))))
                   if seen != made), None)
if repeated != ("observation" in decisions[0]):
    difference = 1  # the first decision is regenerated in the other form
report = {{"verified": difference is None,
          "summary": {{"checked": {{"responses": sum(row["kind"] == "response" for row in tape)}}}}}}
if difference is None:
    report["summary"]["decisions_repeat_observation"] = repeated
else:
    report["error"] = f"Contextual replay differs at record {{difference}}"
json.dump(report, open(arguments["--report"], "w"))
sys.exit(0 if difference is None else 3)
'''
FROM_THE_TAPE = '"observation" in decisions[0]'
FROM_THE_ENVIRONMENT = 'os.environ.get("EXPLORER_RL_DECISION_OBSERVATION", "1") == "1"'
# What the stand-in does on a difference; the tests below put other endings in its place.
REPORTS_THE_DIFFERENCE = '    report["error"] = f"Contextual replay differs at record {difference}"\n'


def capture(folder, *, repeated):
    """A kept game of two seats, cut short while the engine was writing: the last lines are incomplete."""
    observation = {"day": 1, "heroes": []}
    rows = [{"kind": "init", "schema": "heroes-contextual-replay-v1"}]

    def add(**row):
        rows.append({**row, "seq": len(rows) - 1})

    for turn, (player, action) in enumerate(((0, {"kind": "move_to", "x": 1}), (1, {"kind": "end_turn"}),
                                             (0, {"kind": "end_turn"}), (1, {"kind": "end_turn"}))):
        decision = {"kind": "decision", "decision_id": turn, "revision": turn + 1}
        if repeated:
            decision["observation"] = observation
        add(kind="message", player=player, message={"type": "contextual", "records": [
            {"kind": "view", "observation": observation}, decision]})
        add(kind="response", player=player, response={"type": "context_action", "decision_id": turn, "action": action})
    (folder / "profile").mkdir(parents=True)
    (folder / "engine").mkdir()
    (folder / "engine" / "contextual-replay.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows) + '{"kind":"message","pla')
    (folder / "engine" / "replay.jsonl").write_text("".join(json.dumps({"kind": kind}) + "\n" for kind in (
        "init", "start", "input", "applied", "input")))
    return rows


def verifier(tmp_path, rows, form, *, on_difference=REPORTS_THE_DIFFERENCE, game=None):
    original = tmp_path / "original.json"
    original.write_text(json.dumps(rows))
    binary = tmp_path / "bin" / "vcmireplay"
    binary.parent.mkdir(exist_ok=True)
    profile = "VCMI_USER_DATA_DIR" if sys.platform == "darwin" else "XDG_DATA_HOME"
    text = VERIFIER.format(python=sys.executable, original=str(original), form=form, profile=profile,
                           game=str(game or tmp_path / "game" / "profile"))
    assert REPORTS_THE_DIFFERENCE in text
    binary.write_text(text.replace(REPORTS_THE_DIFFERENCE, on_difference))
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    return binary


@pytest.mark.parametrize("repeated", (False, True))
def test_a_capture_of_either_form_passes_with_a_verifier_that_reads_the_form_from_it(tmp_path, monkeypatch, repeated):
    monkeypatch.setenv("VCMI_EXPERIMENT_LEDGER", "0")  # what a shell that launched games may still hold
    monkeypatch.setenv("EXPLORER_RL_DECISION_OBSERVATION", "1" if not repeated else "0")
    rows = capture(tmp_path / "game", repeated=repeated)
    report, passed = check_replay.check(tmp_path / "game", verifier(tmp_path, rows, FROM_THE_TAPE),
                                        tmp_path / "replays", prefix=True)
    assert passed and report["decisions"] == ("repeat the observation" if repeated else "compact")
    assert report["records"] == 9 and report["checked"] == {"responses": 4}  # the unfinished line is left out
    # Seat 0 gave two different answers; one of them took the place of the other.
    assert report["changed_record"] in (1, 5) and "differs at record" in report["changed_answer_failure"]
    assert sorted(item.name for item in (tmp_path / "replays").glob("*contextual*")) == []  # the copies are gone


@pytest.mark.parametrize("repeated", (False, True))
def test_a_verifier_that_takes_the_form_from_its_environment_fails_the_check(tmp_path, repeated):
    rows = capture(tmp_path / "game", repeated=repeated)
    report, passed = check_replay.check(tmp_path / "game", verifier(tmp_path, rows, FROM_THE_ENVIRONMENT),
                                        tmp_path / "replays", prefix=True)
    # A difference at the first decision, before the changed answer, is not a rejection of that answer.
    assert not passed and report["changed_answer_rejected"] is repeated
    # A compact capture fails from a fresh shell; an older one fails once the shell holds the training setting.
    assert report["verified"] is repeated and report["verified_with_the_other_form_in_the_environment"] is False
    assert "differs at record 1" in report["other_environment_failure" if repeated else "plain_failure"]


def test_a_verifier_that_accepts_a_changed_answer_fails_the_check(tmp_path):
    rows = capture(tmp_path / "game", repeated=False)
    binary = verifier(tmp_path, rows, FROM_THE_TAPE)
    binary.write_text(binary.read_text().replace("if seen != made", "if False"))
    report, passed = check_replay.check(tmp_path / "game", binary, tmp_path / "replays", prefix=True)
    assert not passed and report["verified"] and not report["changed_answer_rejected"]


# Ways a verifier can fail on the changed capture without having found the change. Both clean replays pass.
NO_REJECTION = {
    "it crashes before it writes a report": ("    sys.exit(42)\n", "status 42 and no report"),
    "its report is cut off": ('    open(arguments["--report"], "w").write("{"); sys.exit(3)\n', "status 3 and no report"),
    "its report is not a report": ('    json.dump([], open(arguments["--report"], "w")); sys.exit(3)\n', "no report"),
    "it stops for another reason": ('    report["error"] = "Replay initial state ended early"\n', "ended early"),
    "it gives no reason": ("    pass\n", "null"),
    "it differs before the changed answer": ('    report["error"] = "Contextual replay differs at record 0"\n',
                                             "differs at record 0"),
    "it differs but ends with status 0": (REPORTS_THE_DIFFERENCE + "    difference = None\n", "differs at record"),
}


@pytest.mark.parametrize("case", NO_REJECTION)
def test_only_a_reported_difference_after_the_changed_answer_counts_as_its_rejection(tmp_path, case):
    ending, said = NO_REJECTION[case]
    rows = capture(tmp_path / "game", repeated=False)
    binary = verifier(tmp_path, rows, FROM_THE_TAPE, on_difference=ending)
    report, passed = check_replay.check(tmp_path / "game", binary, tmp_path / "replays", prefix=True)
    assert report["verified"] and report["verified_with_the_other_form_in_the_environment"]
    assert not passed and not report["changed_answer_rejected"]
    assert said in report["changed_answer_report"] and "changed_answer_failure" not in report
    assert (tmp_path / "replays" / "changed-answer-console.log").is_file()  # what the verifier printed stays


def test_the_report_of_the_real_verifier_counts():
    """The wording of ``ContextualEvidence::expect``, as the Mac's ``vcmireplay`` gave it on 2026-11-20."""
    real = {"verified": False, "returncode": 3, "error": 'Contextual replay differs at record 3768: expected {\t"kind" : '
            '"message", "message" : {"records" : [{"decision_id" : 133, "kind" : "consumed"}]}}; actual {}'}
    assert check_replay.differs_after(real, 3763)
    assert not check_replay.differs_after(real, 3768) and not check_replay.differs_after({**real, "verified": True}, 3763)


def test_a_failed_check_ends_with_an_error_and_keeps_what_the_verifier_left(fake_engine, tmp_path, monkeypatch, capsys):
    from heroes.config import Config
    engine = fake_engine(days=4)
    config = Config(vcmi=engine.binary, game_data=engine.data, game_map=engine.game_map)
    monkeypatch.setattr("heroes.config.Config", lambda: config)
    state = tmp_path / "state"
    state.mkdir()
    made = []

    def play(config, seed, directory):  # in place of the engine: the game folder of a cut game
        made.append(capture(directory, repeated=False))
        return SimpleNamespace(status="truncated", reason="game_day_limit", detail="", day=3, choices=(2, 2))

    monkeypatch.setattr(check_replay, "play", play)
    rows = capture(tmp_path / "like-the-game", repeated=False)
    binary = verifier(tmp_path, rows, FROM_THE_TAPE, on_difference="    sys.exit(42)\n", game=state)
    monkeypatch.setattr(sys, "argv", ["check_replay.py", str(state), "--replay", str(binary)])
    with pytest.raises(SystemExit, match="The replay check failed"):
        check_replay.main()
    report = json.loads(capsys.readouterr().out)
    kept = Path(report["kept"])
    assert made == [rows] and kept.parent == state and report["changed_answer_rejected"] is False
    assert "status 42 and no report" in report["changed_answer_report"]
    assert (kept / "replays" / "changed-answer-console.log").is_file() and (kept / "game" / "engine").is_dir()


def test_the_check_plays_its_game_with_the_ledger_switched_on(fake_engine, tmp_path, monkeypatch, capsys):
    from heroes import checkpoint
    from heroes.config import Config
    engine = fake_engine(days=4)
    config = Config(vcmi=engine.binary, game_data=engine.data, game_map=engine.game_map)
    monkeypatch.setattr("heroes.config.Config", lambda: config)
    state = tmp_path / "state"
    checkpoint.save(state, checkpoint.fresh(learning_rate=3e-4))
    (engine.binary.parent / "vcmireplay").write_text("")
    monkeypatch.setattr(sys, "argv", ["check_replay.py", str(state), "--max-days", "2"])
    # The scripted engine plays, is stopped at the day limit, and writes no ledger: the check says so.
    with pytest.raises(SystemExit, match="wrote no replay ledger"):
        check_replay.main()
    kept = next(state.glob("check-replay-*"))
    assert str(kept) in capsys.readouterr().err and (kept / "game" / "logs" / "process.log").is_file()
