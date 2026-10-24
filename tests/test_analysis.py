"""Offline episode analysis over synthetic run directories, never VCMI/game evidence."""
from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("heroes_analysis", ROOT / "experiments/analysis/analyze_runs.py")
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)
FIXTURE = ROOT / "tests/fixtures/analysis/synthetic-verified-loss"


def tree_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if item.is_file():
            digest.update(str(item.relative_to(path)).encode() + item.read_bytes())
    return digest.hexdigest()


class SyntheticRuns:
    """Copies the labelled synthetic fixture and derives variants; the original is never edited."""

    def __init__(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.base = json.loads((FIXTURE / "manifest.json").read_text())

    def run(self, name: str, update=None, ledger: list[dict] | None = None, manifest_text: str | None = None) -> str:
        path = self.root / name
        shutil.copytree(FIXTURE, path)
        manifest = copy.deepcopy(self.base)
        if update:
            update(manifest)
        if ledger is not None:
            (path / "policy.jsonl").write_text("".join(json.dumps(entry) + "\n" for entry in ledger))
            manifest["policy_ledger"]["sha256"] = hashlib.sha256((path / "policy.jsonl").read_bytes()).hexdigest()
        (path / "manifest.json").write_text(manifest_text if manifest_text is not None else json.dumps(manifest))
        return str(path)

    def close(self):
        self.directory.cleanup()

    def win(self, name: str) -> str:
        ledger = [json.loads(line) for line in (FIXTURE / "policy.jsonl").read_text().splitlines()]
        for entry in ledger:
            if entry["message"].get("type") == "terminal":
                entry["message"]["outcome"] = "win"
        return self.run(name, win, ledger=ledger)


def win(manifest):
    manifest.update(outcome="win", reward=1)
    manifest["policy_execution"]["terminal"]["outcome"] = "win"


def crash(manifest):
    # Mirrors attempt 2: socket loss, then an aborting process. Must stay unscored.
    manifest.update(status="failed", outcome=None, reward=None, returncode=-6,
                    termination_reason="process_exit_without_verified_terminal")


def timeout(manifest):
    # Mirrors attempt 4: socket loss, then a hung shutdown and a wall-time kill.
    manifest.update(status="timeout", outcome=None, reward=None, returncode=-9, termination_reason="wall_time_limit")


def state(day, heroes, towns):
    return {"day": day, "own_heroes": heroes, "own_towns": towns, "own_mines": 0, "visible_tiles": 10}


def engine(elapsed, message_type, **fields):
    return {"elapsed_seconds": elapsed, "direction": "engine",
            "message": {"type": message_type, "player": 0, "episode": "1", **fields}}


class AnalysisTestCase(unittest.TestCase):
    def setUp(self):
        self.runs = SyntheticRuns()

    def tearDown(self):
        self.runs.close()

    def analyze(self, *runs):
        return analysis.analyze(list(runs))


class OutcomeVerificationTests(AnalysisTestCase):
    def test_valid_win_and_loss_are_verified_and_scored(self):
        report = self.analyze(self.runs.run("loss"), self.runs.win("win"))
        loss, victory = report["attempts"]
        self.assertEqual((loss["classification"], loss["scored_outcome"], loss["scored_reward"]),
                         ("verified_natural", "loss", -1))
        self.assertEqual((victory["classification"], victory["scored_outcome"], victory["scored_reward"]),
                         ("verified_natural", "win", 1))
        self.assertTrue(all(check["passed"] for check in loss["verification"]))
        self.assertTrue(all(check["passed"] for check in victory["verification"]))
        self.assertEqual(victory["ledger"]["terminal_outcome"], "win")
        self.assertTrue(loss["synthetic_fixture"])
        summary = report["summary"]
        self.assertEqual(len(summary["cohorts"]), 1)
        self.assertEqual(summary["overall"]["verified_outcomes"], {"win": 1, "loss": 1})
        self.assertEqual(summary["overall"]["natural_completion"]["denominator"], 2)
        self.assertEqual(summary["overall"]["natural_completion"]["frequency"], 1.0)

    def test_crash_or_timeout_with_reported_terminal_loss_is_never_a_loss(self):
        report = self.analyze(self.runs.run("crash", crash), self.runs.run("timeout", timeout))
        crashed, timed_out = report["attempts"]
        self.assertEqual(crashed["classification"], "technical_failure")
        self.assertEqual(timed_out["classification"], "truncated")
        for attempt in (crashed, timed_out):
            self.assertIsNone(attempt["scored_outcome"])
            self.assertIsNone(attempt["scored_reward"])
            self.assertEqual(attempt["reported_terminal_outcome"], "loss")
            self.assertTrue(attempt["reported_terminal_unscored"])
        overall = report["summary"]["overall"]
        self.assertEqual(overall["verified_outcomes"], {"win": 0, "loss": 0})
        self.assertEqual(overall["natural_completion"]["verified_natural"], 0)
        self.assertEqual(overall["natural_completion"]["denominator"], 2)
        self.assertEqual(overall["truncations_by_reason"], {"wall_time_limit": 1})
        self.assertEqual(overall["technical_failures_by_reason"], {"process_exit_without_verified_terminal": 1})
        self.assertIn("loss (unscored)", analysis.markdown(report))

    def test_inconsistent_evidence_is_unscored(self):
        def socket_disagrees(m):
            m["policy_execution"]["terminal"]["outcome"] = "win"

        def wrong_reward(m):
            m["reward"] = 1

        def bool_reward(m):
            m["reward"] = True
            m.update(outcome="win")
            m["policy_execution"]["terminal"]["outcome"] = "win"

        def unclean_exit(m):
            m["returncode"] = -11

        def invalid_recording(m):
            m["recordings"]["player"]["recording_valid"] = False

        def never_ready(m):
            m["policy_execution"]["ready"] = False

        def limit_hit(m):
            m["policy_execution"]["limit"] = "agent_action_limit"

        def policy_error(m):
            m["policy_execution"]["error"] = "late protocol failure"

        def missing_reward(m):
            del m["reward"]

        def timeout_with_reward(m):
            timeout(m)
            m["reward"] = -1

        variants = (socket_disagrees, wrong_reward, bool_reward, unclean_exit, invalid_recording,
                    never_ready, limit_hit, policy_error, missing_reward, timeout_with_reward)
        report = self.analyze(*(self.runs.run(v.__name__, v) for v in variants))
        for variant, attempt in zip(variants, report["attempts"]):
            with self.subTest(variant=variant.__name__):
                self.assertEqual(attempt["classification"], "inconsistent")
                self.assertIsNone(attempt["scored_outcome"])
                self.assertIsNone(attempt["scored_reward"])
        self.assertEqual(report["summary"]["overall"]["verified_outcomes"], {"win": 0, "loss": 0})

    def test_legacy_nested_terminal_reward_is_ignored(self):
        def legacy_positive(m):
            m["policy_execution"]["terminal"]["reward"] = 1

        def legacy_on_timeout(m):
            timeout(m)
            m["policy_execution"]["terminal"]["reward"] = -1

        report = self.analyze(self.runs.run("legacy", legacy_positive), self.runs.run("legacy-timeout", legacy_on_timeout))
        verified, truncated = report["attempts"]
        self.assertEqual((verified["classification"], verified["scored_reward"]), ("verified_natural", -1))
        self.assertEqual((truncated["classification"], truncated["scored_reward"]), ("truncated", None))
        for attempt in report["attempts"]:
            self.assertIn("legacy_terminal_reward_ignored", [d["code"] for d in attempt["diagnostics"]])

    def test_verified_ledger_terminal_contradiction_is_unscored(self):
        # Deliberately change only the manifest; its declared ledger still reports loss.
        attempt = self.analyze(self.runs.run("contradictory", win))["attempts"][0]
        self.assertTrue(attempt["ledger"]["hash_verified"])
        self.assertTrue(attempt["ledger"]["used_for_metrics"])
        self.assertEqual(attempt["classification"], "inconsistent")
        self.assertIsNone(attempt["scored_outcome"])
        self.assertIsNone(attempt["scored_reward"])
        self.assertIn("ledger_terminal_mismatch", [d["code"] for d in attempt["diagnostics"]])
        self.assertFalse(next(c["passed"] for c in attempt["verification"]
                              if c["check"] == "ledger_terminal_matches"))


class InputValidityTests(AnalysisTestCase):
    def test_wrongly_typed_scalar_fields_do_not_abort_other_attempts(self):
        variants = [("schema_version", []), ("kind", {}), ("schema_version", 2.0),
                    ("outcome", {}), ("outcome", ["loss"])]
        paths = [self.runs.run(f"bad-{index}", lambda m, k=key, v=value: m.update({k: v}))
                 for index, (key, value) in enumerate(variants)]
        report = self.analyze(*paths, self.runs.run("good"))
        self.assertEqual(len(report["attempts"]), len(paths) + 1)
        self.assertEqual([a["validity"] for a in report["attempts"][:3]], ["malformed"] * 3)
        self.assertEqual([a["classification"] for a in report["attempts"][3:5]], ["inconsistent"] * 2)
        self.assertEqual(report["attempts"][-1]["scored_reward"], -1)
        self.assertIn("invalid: malformed", analysis.markdown(report))

    def test_every_requested_attempt_is_listed_with_diagnostics(self):
        empty = self.runs.root / "empty"
        empty.mkdir()

        def schema_v1(m):
            m["schema_version"] = 1

        def smoke(m):
            m["kind"] = "recorder-smoke"

        def no_status(m):
            del m["status"]

        good = self.runs.run("good")
        inputs = [good, str(self.runs.root / "absent"), str(empty), self.runs.run("bad-json", manifest_text="{oops"),
                  self.runs.run("array", manifest_text="[]"), self.runs.run("v1", schema_v1),
                  self.runs.run("smoke", smoke), self.runs.run("no-status", no_status), good]
        report = self.analyze(*inputs)
        validity = [(a["validity"], a["diagnostics"][0]["code"] if a["validity"] != "eligible" else None)
                    for a in report["attempts"]]
        self.assertEqual(validity, [("eligible", None), ("missing", "run_missing"), ("missing", "manifest_missing"),
                                    ("malformed", "manifest_not_json"), ("malformed", "manifest_not_object"),
                                    ("unsupported", "manifest_unsupported"), ("unsupported", "manifest_unsupported"),
                                    ("malformed", "status_missing"), ("duplicate", "duplicate_input")])
        summary = report["summary"]
        self.assertEqual((summary["supplied"], summary["eligible"], summary["invalid"]), (9, 1, 8))
        self.assertEqual(summary["invalid_by_reason"],
                         {"duplicate": 1, "malformed": 3, "missing": 2, "unsupported": 2})
        self.assertEqual(summary["overall"]["natural_completion"]["denominator"], 1)
        text = analysis.markdown(report)
        for number in range(1, 10):
            self.assertIn(f"| {number} |", text)

    def test_wrongly_typed_sections_and_missing_metrics_stay_unknown(self):
        def damaged(m):
            m["policy_execution"] = "not an object"
            del m["map"]
            del m["policy"]["seed"]

        attempt = self.analyze(self.runs.run("damaged", damaged))["attempts"][0]
        self.assertEqual(attempt["validity"], "eligible")
        self.assertEqual(attempt["classification"], "inconsistent")
        self.assertIn("map.name", attempt["unknown_metadata"])
        self.assertEqual(attempt["seeds"]["policy_seed"], "unknown")
        self.assertIsNone(attempt["metrics"]["actions_sent"])
        self.assertIn("actions_sent", attempt["metrics"]["missing"])
        self.assertIn("policy_execution_invalid", [d["code"] for d in attempt["diagnostics"]])


class CohortAndSeedTests(AnalysisTestCase):
    def test_incompatible_cohorts_are_not_pooled(self):
        def other_map(m):
            m["map"]["sha256"] = "9" * 64

        def blue_side(m):
            m.update(agent_player_index=1, outcome_player="blue")

        def other_limits(m):
            m["limits"]["max_days"] = 10

        def other_engine(m):
            m["engine_recipe"]["engine/patches/0001-experiment-environment.patch"] = "8" * 64

        def checkpoint(m):
            m["policy"]["checkpoint"] = "synthetic-checkpoint-1"

        def unknown_opponent(m):
            del m["opponent"]

        variants = (other_map, blue_side, other_limits, other_engine, checkpoint, unknown_opponent)
        report = self.analyze(self.runs.run("base"), self.runs.win("base-win"),
                              *(self.runs.run(v.__name__, v) for v in variants))
        cohorts = report["summary"]["cohorts"]
        self.assertEqual(len(cohorts), 1 + len(variants))
        self.assertEqual(cohorts[0]["attempts"], [1, 2])
        self.assertEqual(cohorts[0]["verified_outcomes"], {"win": 1, "loss": 1})
        self.assertIsNone(report["summary"]["overall"]["verified_outcomes"])
        self.assertIn("Not pooled", report["summary"]["overall"]["verified_outcomes_note"])
        unknown = cohorts[-1]
        self.assertEqual(unknown["components"]["opponent"]["module"], "unknown")
        self.assertIn("opponent.module", unknown["unknown_metadata"])

    def test_policy_delay_and_capture_cadence_split_bounded_cohorts(self):
        def slower_policy(m):
            m["policy"]["step_delay_seconds"] = 5.0

        def slower_capture(m):
            for view in m["recordings"].values():
                view["capture"]["fps"] = 1

        report = self.analyze(self.runs.run("base"), self.runs.run("slow-policy", slower_policy),
                              self.runs.run("slow-capture", slower_capture))
        self.assertEqual(len(report["summary"]["cohorts"]), 3)
        self.assertIsNone(report["summary"]["overall"]["verified_outcomes"])

    def test_repeated_seed_pairs_are_flagged_and_retained(self):
        def second_seed(m):
            m["seed"] = 12346
            m["effective_engine_rng"]["initial_seed"] = 12346
            m["policy"]["seed"] = 2

        def logged_seed_differs(m):
            m["effective_engine_rng"]["initial_seed"] = 999

        report = self.analyze(self.runs.run("a"), self.runs.run("b", timeout), self.runs.run("c", second_seed),
                              self.runs.run("d", logged_seed_differs))
        repeated = report["summary"]["repeated_seed_pairs"]
        self.assertEqual(len(repeated), 1)
        self.assertEqual((repeated[0]["engine_seed"], repeated[0]["policy_seed"], repeated[0]["attempts"]),
                         (12345, 1, [1, 2, 4]))
        self.assertEqual(repeated[0]["classifications"], ["verified_natural", "truncated", "verified_natural"])
        self.assertEqual([a["seeds"]["engine_seed"] for a in report["attempts"]], [12345, 12345, 12346, 12345])
        self.assertIn("engine_seed_mismatch", [d["code"] for d in report["attempts"][3]["diagnostics"]])


class LedgerAndMetricTests(AnalysisTestCase):
    def test_malformed_nested_ledger_fields_withhold_metrics_without_aborting_batch(self):
        variants = []
        for value in ([], {}, None):
            variants.append(engine(0.1, "terminal", outcome="loss", events=[{"kind": value}]))
        variants += [engine(0.1, {}, outcome="loss"), engine(0.1, "terminal", outcome=[]),
                     {"elapsed_seconds": 0.1, "direction": [], "message": {}}]
        paths = [self.runs.run(f"bad-ledger-{index}", ledger=[entry])
                 for index, entry in enumerate(variants)]
        report = self.analyze(*paths, self.runs.run("good"))
        for attempt in report["attempts"][:-1]:
            self.assertIn("ledger_malformed", [d["code"] for d in attempt["diagnostics"]])
            self.assertFalse(attempt["ledger"]["used_for_metrics"])
            self.assertIsNone(attempt["metrics"]["ledger_event_counts"])
        self.assertTrue(report["attempts"][-1]["ledger"]["used_for_metrics"])

    def test_ledger_identity_and_timestamp_errors_withhold_metrics(self):
        wrong_player = engine(0.1, "observation", state=state(1, 0, 1))
        wrong_player["message"]["player"] = 1
        wrong_episode = engine(0.1, "observation", state=state(1, 0, 1))
        wrong_episode["message"]["episode"] = "another-episode"
        backwards = [engine(10.0, "observation", state=state(1, 0, 1)),
                     engine(1.0, "observation", state=state(2, 0, 1))]
        negative = [engine(-0.1, "observation", state=state(1, 0, 1))]
        variants = [([wrong_player], "ledger_identity_mismatch"), ([wrong_episode], "ledger_identity_mismatch"),
                    (backwards, "ledger_time_invalid"), (negative, "ledger_time_invalid")]
        report = self.analyze(*(self.runs.run(f"invalid-{index}", ledger=ledger)
                                for index, (ledger, _) in enumerate(variants)))
        for attempt, (_, code) in zip(report["attempts"], variants):
            self.assertIn(code, [d["code"] for d in attempt["diagnostics"]])
            self.assertTrue(attempt["ledger"]["hash_verified"])
            self.assertFalse(attempt["ledger"]["used_for_metrics"])
            self.assertNotIn("town_only_intervals", attempt["ledger"])

    def test_metrics_are_surfaced_from_recorded_evidence(self):
        attempt = self.analyze(self.runs.run("loss"))["attempts"][0]
        m = attempt["metrics"]
        self.assertEqual((m["actions_sent"], m["actions_rejected"], m["successful_displacements"],
                          m["unique_hero_positions"]), (2, 0, 1, 2))
        self.assertEqual(m["visible_tiles"], {"initial": 10, "final": 15, "final_source": "terminal.state",
                                              "expansion": 5})
        self.assertEqual(m["game_days"], {"initial_observed": 1, "last_observed": 1, "terminal": 2})
        self.assertEqual(m["completed_battles"], {"value": 1, "source": "terminal.state"})
        self.assertEqual(m["own_mines"]["max_observed_in_ledger"], 0)
        self.assertEqual(m["ledger_event_counts"], {"battle_complete": 1, "visit": 1})
        self.assertTrue(attempt["ledger"]["hash_verified"])

    def test_town_only_survival_is_an_observed_interval_not_a_defeat(self):
        ledger = [engine(0.1, "observation", action_id=0, day=1, state=state(1, 1, 1)),
                  engine(0.2, "action_result", action_id=0, action=0, accepted=True,
                         before=state(1, 1, 1), after=state(1, 0, 1), events=[{"kind": "battle_complete"}]),
                  engine(0.3, "observation", action_id=1, day=1, state=state(1, 0, 1)),
                  engine(0.4, "action_result", action_id=1, action=8, accepted=True,
                         before=state(1, 0, 1), after=state(1, 0, 1), events=[]),
                  engine(0.5, "observation", action_id=2, day=2, state=state(2, 0, 1))]

        def wall_timeout(m):
            timeout(m)
            m["policy_execution"]["terminal"] = None

        report = self.analyze(self.runs.run("town-only", wall_timeout, ledger=ledger))
        attempt = report["attempts"][0]
        self.assertEqual(attempt["classification"], "truncated")
        self.assertIsNone(attempt["scored_outcome"])
        interval = attempt["ledger"]["town_only_intervals"]
        self.assertEqual(len(interval), 1)
        interval = interval[0]
        self.assertEqual(interval["preceding_observation"]["elapsed_seconds"], 0.2)
        self.assertEqual(interval["preceding_observation"]["source"], "action_result.before")
        self.assertEqual((interval["first_observed"]["elapsed_seconds"], interval["first_observed"]["source"]),
                         (0.2, "action_result.after"))
        self.assertEqual(interval["last_observed"]["day"], 2)
        self.assertTrue(interval["right_censored"])
        self.assertFalse(interval["left_censored"])
        self.assertEqual(report["summary"]["overall"]["verified_outcomes"], {"win": 0, "loss": 0})
        self.assertIn("right-censored at ledger end", analysis.markdown(report))

    def test_town_only_interval_ended_by_observed_town_loss(self):
        ledger = [engine(0.1, "observation", action_id=0, day=1, state=state(1, 0, 1)),
                  engine(0.9, "terminal", outcome="loss", state=state(3, 0, 0), events=[])]
        attempt = self.analyze(self.runs.run("ended", ledger=ledger))["attempts"][0]
        self.assertEqual(attempt["classification"], "verified_natural")
        town_only = attempt["ledger"]["town_only_intervals"][0]
        self.assertTrue(town_only["left_censored"])
        self.assertFalse(town_only["right_censored"])
        self.assertEqual(town_only["following_observation"]["own_towns"], 0)
        self.assertEqual(len(attempt["ledger"]["zero_hero_intervals"]), 1)

    def test_ledger_hash_mismatch_and_malformed_ledgers_withhold_ledger_metrics(self):
        tampered = Path(self.runs.run("tampered"))
        with (tampered / "policy.jsonl").open("a") as ledger:
            ledger.write("{}\n")
        malformed = self.runs.run("malformed", ledger=[{"direction": "engine", "message": {}}])
        absent = Path(self.runs.run("absent"))
        (absent / "policy.jsonl").unlink()
        report = self.analyze(str(tampered), malformed, str(absent))
        for attempt, code in zip(report["attempts"], ("ledger_hash_mismatch", "ledger_malformed", "ledger_missing")):
            with self.subTest(code=code):
                self.assertIn(code, [d["code"] for d in attempt["diagnostics"]])
                self.assertFalse(attempt["ledger"]["used_for_metrics"])
                self.assertIsNone(attempt["metrics"]["own_mines"]["max_observed_in_ledger"])
                # Manifest-level verification is independent of the supplementary ledger.
                self.assertEqual(attempt["classification"], "verified_natural")
        self.assertFalse(report["attempts"][0]["ledger"]["hash_verified"])


class CommandLineTests(AnalysisTestCase):
    def test_cli_writes_new_reports_and_leaves_sources_untouched(self):
        runs = [self.runs.run("loss"), self.runs.run("timeout", timeout), str(self.runs.root / "absent")]
        before = {run: tree_digest(Path(run)) for run in runs[:2]}
        fixture_before = tree_digest(FIXTURE)
        out = self.runs.root / "reports"
        self.assertEqual(analysis.main([*runs, "--json", str(out / "a.json"), "--markdown", str(out / "a.md")]), 0)
        report = json.loads((out / "a.json").read_text())
        self.assertEqual(report["kind"], "episode-analysis")
        self.assertEqual(report["summary"]["supplied"], 3)
        text = (out / "a.md").read_text()
        self.assertIn("Synthetic fixtures present", text)
        self.assertIn("does not establish improvement", text)
        self.assertEqual({run: tree_digest(Path(run)) for run in runs[:2]}, before)
        self.assertEqual(tree_digest(FIXTURE), fixture_before)
        for argv in ([*runs, "--json", str(out / "a.json")],
                     [runs[0], "--markdown", str(Path(runs[0]) / "report.md")]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                analysis.main(argv)
        self.assertFalse((Path(runs[0]) / "report.md").exists())

    def test_cli_prints_json_without_output_files(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(analysis.main([self.runs.run("loss")]), 0)
        self.assertEqual(json.loads(stdout.getvalue())["summary"]["eligible"], 1)


if __name__ == "__main__":
    unittest.main()
