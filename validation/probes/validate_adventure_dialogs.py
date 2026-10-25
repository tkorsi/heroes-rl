"""Bounded native dialog controls on isolated diagnostic maps, never training.

The schedule and seeds are written before launch. Choices use the delivered
dialog only; authored fixture metadata is restricted to effect checks. Missing
random branches are retained as unverified, never replaced by injected results.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.explorer.create_dialog_scenarios import MAPS, scenario_id, verify_dialog_scenario
from experiments.headless import env as environment
from experiments.headless.contextual_evidence import runtime_identity
from experiments.learner import contextual as c
from experiments.learner.first_run import implementation_identity
from experiments.learner.inference import digest, write_json
from validation.probes import audit_contextual_histories as auditor
from validation.probes import validate_contextual_native as base
from validation.probes.validate_army_management import native_receipt
from validation.probes.validate_hero_policy import retain_failed_evidence, wall_limit
from validation.probes.validate_hero_recruitment import HeroRun

INTERFACE = "adventure-dialogs-v1"
ACTION_SCHEMA = "heroes-context-actions-v7"
CASES = {
    "mine-accept": ("guarded_mine", "accept"), "mine-decline": ("guarded_mine", "decline"),
    "abandoned-accept": ("abandoned_mine", "accept"), "abandoned-decline": ("abandoned_mine", "decline"),
    "free-accept": ("neutral_free", "accept"),
    "free-decline-pursue": ("neutral_free", "pursue"),
    "free-decline-release": ("neutral_free", "let_flee"),
    "flee-pursue": ("neutral_flee", "pursue"), "flee-release": ("neutral_flee", "let_flee"),
    "paid-accept": ("neutral_paid", "accept"), "paid-decline": ("neutral_paid", "decline"),
    "chest": ("chest", None), "arena": ("arena", None),
    "school-magic": ("school_magic", None), "school-war": ("school_war", None),
    "scholar": ("scholar", None), "scholar-exchange": ("scholar_exchange", None),
    "free-decline-attack": ("neutral_free_attack", "decline_attack"),
}
OBJECT_KINDS = {"guarded_mine": 53, "abandoned_mine": 53, "neutral_free": 54,
                "neutral_flee": 54, "neutral_paid": 54, "chest": 101, "arena": 4,
                "school_magic": 47, "school_war": 107, "scholar": 81, "neutral_free_attack": 54}


def dialog_open_proof(parent, after, additions, result):
    """Bind the exact delivered payload, parent receipt and fresh decision."""
    context = after.observation.context
    panel = after.observation.window.dialog_panel
    base.ensure(context.kind == "dialog" and panel is not None and panel.ref == context.selection,
                "Expected the actual delivered dialog context")
    requests = [frame for frame in additions if frame.event is not None
                and frame.event.kind == "dialog_requested" and frame.event.dialog == panel
                and frame.event.subject == context.subject]
    endings = [frame for frame in additions if frame.kind == "result" and frame.decision_seq == parent.seq
               and frame.result == result]
    opened = [frame for frame in additions if frame.kind == "context_opened"
              and frame.context == context and frame.reason == "adventure_dialog"]
    base.ensure(len(requests) == len(endings) == len(opened) == 1
                and requests[0].seq < endings[0].seq < opened[0].seq < after.seq,
                "Dialog payload, parent completion and context opening are missing or reordered")
    base.ensure(result.accepted and result.status in ("interaction", "dialog_answered"),
                "Dialog parent did not complete as a native interaction or answer")
    base.ensure(after.observation.window.armies == () and after.observation.window.army is None
                and after.observation.window.stack_panel is None,
                "Mandatory dialog exposed private parent army fields")
    return {"family": panel.family, "dialog_ref": panel.ref, "request_seq": requests[0].seq,
            "parent_result_seq": endings[0].seq, "opened_seq": opened[0].seq, "decision_seq": after.seq}


def explicit_choice(panel, kind, component_index=None):
    matches = [choice for choice in panel.choices
               if choice.kind == kind and choice.component_index == component_index]
    base.ensure(len(matches) == 1, "Requested diagnostic choice is not uniquely displayed")
    return matches[0]


def exact_resource_delta(before, after, gold):
    delta = tuple(new - old for old, new in zip(before, after))
    base.ensure(delta == (0, 0, 0, 0, 0, 0, gold), "Native dialog changed resources beyond its displayed gold effect")
    return list(delta)


def notice_templates(additions):
    return [token.text_id for frame in additions if frame.event is not None
            and frame.event.kind == "notice" for token in frame.event.notice.text if token.op == "append_text"]


def army_counts(window):
    result = {}
    for stack in window.army:
        result[stack.creature] = result.get(stack.creature, 0) + stack.count
    return result


def refusal_attack_proof(additions):
    notice = [frame.seq for frame in additions if frame.event is not None and frame.event.kind == "notice"
              and any(token.text_id == "core.advevent.87" for token in frame.event.notice.text)]
    starts = [frame.seq for frame in additions if frame.event is not None and frame.event.kind == "battle_start"]
    ends = [frame.seq for frame in additions if frame.event is not None and frame.event.kind == "battle_complete"]
    base.ensure(len(notice) == len(starts) == len(ends) == 1 and notice[0] < starts[0] < ends[0],
                "Native free-join refusal lacks its ordered ADVOB87 notice and completed battle")
    return {"refusal_notice_seq": notice[0], "battle_start_seq": starts[0], "battle_complete_seq": ends[0]}


class DialogRun(HeroRun):
    def expected_offers(self, frame):
        return c.Offers.from_observation(frame.observation, schema=ACTION_SCHEMA)

    def action(self, kind, **parameters):
        return c.Action(kind, schema=ACTION_SCHEMA, **parameters)

    def live(self):
        # A mandatory query has no neutral cutoff choice. Exhaustion is a
        # technical diagnostic failure, not an invented strategy or draw.
        if self.state is None or self.state.done:
            return False
        if self.commands >= self.args.max_decisions or time.monotonic() - self.started > self.args.case_seconds:
            if self.inspect_request().observation.context.kind == "dialog":
                raise base.ProbeFailure("Declared diagnostic bound reached during a mandatory dialog")
            return super().live()
        return True

    def own_hero(self):
        observation = self.inspect_request().observation
        base.ensure(observation.heroes, "Diagnostic actor has no remaining native hero")
        if not hasattr(self, "actor"):
            town = observation.towns[0]
            self.actor = next((hero.handle for hero in observation.heroes if hero.position == town.position),
                              observation.heroes[0].handle)
        base.ensure(any(hero.handle == self.actor for hero in observation.heroes), "Diagnostic actor was defeated")
        return self.actor

    def hero_window(self):
        self.call(self.action("open_hero", subject=self.own_hero()), "explicit_inspect_own_hero")
        window = self.inspect_request().observation.window
        base.ensure(window.context.kind == "hero", "Own hero inspection did not open its native window")
        self.call(self.action("close"), "explicit_close_own_hero")
        return window

    def targets(self, kind, visited=()):
        observation = self.inspect_request().observation
        return sorted((obj for obj in observation.objects if obj.kind == kind and obj.handle not in visited),
                      key=lambda obj: obj.handle)

    def visit(self, target):
        combined = []
        for _ in range(12):
            base.ensure(self.live(), "Native movement exhausted the predeclared diagnostic bounds")
            frame = self.inspect_request()
            base.ensure(frame.observation.context.kind == "map", "Object visit requires the map")
            current = next((obj for obj in frame.observation.objects if obj.handle == target.handle), None)
            base.ensure(current is not None and current.position == target.position,
                        "Diagnostic destination is no longer currently visible")
            hero = next(hero for hero in frame.observation.heroes if hero.handle == self.own_hero())
            if not hero.movement:
                self.next_own_day(self.state.player, frame.day)
                continue
            before, additions, result = self.call(self.action("move_to", subject=hero.handle,
                x=current.position[0], y=current.position[1]), "explicit_visit_currently_visible_fixture")
            combined.extend(additions)
            base.ensure(not self.state.done, "Native fixture visit unexpectedly ended the game")
            after = self.inspect_request()
            if after.observation.context.kind == "dialog":
                self.check("native_dialog_open_order", "passed", **dialog_open_proof(before, after, additions, result))
                return combined
            if after.observation.context.kind == "exchange":
                return combined
            base.ensure(result is not None and result.accepted, "Fixture movement lacks an accepted native result")
            if result.status == "interaction" or any(f.event is not None and f.event.kind == "notice" for f in additions):
                return combined
            if result.status == "reached":
                return combined
            base.ensure(result.status not in ("no_route_known", "rejected", "interrupted"),
                        "Visible fixture could not be reached using the native route")
            self.next_own_day(self.state.player, after.day)
        raise base.ProbeFailure("Visible-object visit exhausted twelve bounded route attempts")

    def answer(self, kind, component_index=None):
        before = self.inspect_request()
        base.ensure(before.observation.context.kind == "dialog", "Answer requires a delivered mandatory query")
        panel = before.observation.window.dialog_panel
        choice = explicit_choice(panel, kind, component_index)
        self.reject(self.action("choose_dialog", subject=self.own_hero(), selection="not-current-dialog-choice"),
                    "stale_or_foreign_dialog_choice_rejected_before_send")
        _, additions, result = self.call(self.action("choose_dialog", subject=before.observation.context.subject,
            selection=choice.ref), "explicit_dialog_" + kind)
        proof = native_receipt(before, additions, result, "dialog_answered")
        base.ensure(not self.state.done, "Diagnostic answer unexpectedly ended the native episode")
        after = self.inspect_request()
        if after.observation.context.kind == "dialog":
            self.check("native_chained_dialog_order", "passed", **dialog_open_proof(before, after, additions, result))
        self.check("explicit_native_dialog_answer", "passed", family=panel.family,
                   choice=asdict(choice), displayed=asdict(panel), **proof)
        return before, after, additions, choice

    def revisit(self, target):
        self.depart(self.own_hero(), "explicit_departure_before_revisit")
        return self.visit(target)

    def family(self):
        observation = self.inspect_request().observation
        return observation.window.dialog_panel.family if observation.context.kind == "dialog" else None


def mine_case(run, action):
    target = run.targets(53)[0]
    run.visit(target)
    base.ensure(run.family() == "guarded_mine", "Native guarded mine did not ask its actual yes/no question")
    before, after, additions, _ = run.answer(action)
    exact_resource_delta(before.observation.resources, after.observation.resources, 0)
    mine = next(obj for obj in after.observation.objects if obj.handle == target.handle)
    battle = any(f.event is not None and f.event.kind == "battle_start" for f in additions)
    base.ensure((action == "accept") == battle and mine.owner == ("self" if action == "accept" else "neutral"),
                "Native mine answer did not produce its expected battle/ownership branch")
    run.check("native_mine_answer_effect", "passed", answer=action, battle=battle, owner=mine.owner,
              accepted_native_reply=True, native_rejected_ack=False)


def neutral_case(run, fixture, strategy):
    previous = run.hero_window()
    target = run.targets(54)[0]
    run.visit(target)
    expected = {"neutral_free": "neutral_free_join", "neutral_free_attack": "neutral_free_join", "neutral_paid": "neutral_paid_join",
                "neutral_flee": "neutral_flee"}[fixture]
    initial = run.family()
    if initial != expected:
        run.check("requested_native_random_branch", "unverified", expected=expected, observed=initial,
                  reason="Predeclared native seed produced another ordinary branch")
    additions = []
    families = []
    selected = None
    for _ in range(3):
        family = run.family()
        if family is None:
            break
        families.append(family)
        if family == "neutral_flee":
            choice = strategy if strategy in ("pursue", "let_flee") else "let_flee"
        else:
            base.ensure(family in ("neutral_free_join", "neutral_paid_join"), "Unexpected neutral query family")
            choice = "accept" if strategy == "accept" else "decline"
        before, after, more, selected = run.answer(choice)
        additions.extend(more)
        if choice == "accept" and family in ("neutral_free_join", "neutral_paid_join"):
            cost = 0 if family == "neutral_free_join" else next(
                component.amount for component in before.observation.window.dialog_panel.components if component.kind == "resource")
            if before.observation.resources[6] >= cost:
                exact_resource_delta(before.observation.resources, after.observation.resources, -cost)
                current = run.hero_window()
                old, new = army_counts(previous), army_counts(current)
                # Fixture count is an oracle only; no hidden count enters the
                # choice or the dialog/model observation for free joining.
                expected_army = dict(old); expected_army[0] = expected_army.get(0, 0) + 6
                base.ensure(new == expected_army, "Native joining did not add the fixture's six Pikemen exactly")
                run.check("native_join_exact_effect", "passed", family=family, gold_debit=cost,
                          added_pikemen=6, hidden_fixture_count_used_for_choice=False)
            else:
                base.ensure("core.genrltxt.29" in notice_templates(more), "Unaffordable native reply omitted its visible notice")
    base.ensure(run.family() is None, "Neutral query chain exceeded three explicit answers")
    if fixture == "neutral_free" and strategy in ("pursue", "let_flee"):
        base.ensure(families == ["neutral_free_join", "neutral_flee"],
                    "Free-join refusal did not produce the required real chained flee decision")
    if fixture == "neutral_free_attack":
        base.ensure(families == ["neutral_free_join"], "Never-flee fixture produced another strategic query")
        proof = refusal_attack_proof(additions)
        base.ensure(any(hero.handle == run.actor for hero in run.inspect_request().observation.heroes),
                    "Diagnostic actor did not survive the native refusal battle")
        run.check("native_free_refusal_attack", "passed", **proof)
    if strategy in ("pursue", "let_flee"):
        battle = any(f.event is not None and f.event.kind == "battle_start" for f in additions)
        base.ensure(battle == (strategy == "pursue"), "Neutral flee answer produced the wrong battle branch")
    run.check("native_neutral_branch_summary", "passed", initial_family=initial,
              delivered_families=families,
              observed_battle=any(f.event is not None and f.event.kind == "battle_start" for f in additions),
              target_still_visible=any(obj.handle == target.handle for obj in run.inspect_request().observation.objects))


def rewardable_case(run, fixture):
    visited = set(); seen = set(); selection = 0; first = True
    kind = OBJECT_KINDS[fixture]
    for _ in range(9):
        targets = run.targets(kind, visited)
        if not targets:
            break
        target = targets[0]
        prior = run.hero_window()
        additions = run.visit(target)
        family = run.family()
        if family is None:
            templates = notice_templates(additions)
            seen.update(templates)
            if fixture == "scholar":
                notices = [f.event.notice for f in additions if f.event is not None and f.event.kind == "notice"
                           and any(t.text_id == "core.advevent.115" for t in f.event.notice.text)]
                base.ensure(len(notices) == 1 and len(notices[0].text) == 1 and len(notices[0].components) == 1,
                            "Scholar exposed unused replacements or omitted its actual one-component notice")
                seen.add("scholar_" + notices[0].components[0].kind)
                run.check("native_scholar_displayed_only_notice", "passed", notice=asdict(notices[0]))
            elif fixture == "chest":
                base.ensure("core.advevent.145" in templates, "Chest neither offered a choice nor displayed its native artifact")
                run.check("native_chest_artifact_notice", "passed", templates=templates)
            else:
                expected = "core.advevent.73" if fixture == "school_magic" else "core.advevent.160"
                base.ensure(expected in templates, "School neither offered its choice nor showed insufficient gold")
                run.check("native_school_insufficient_funds", "passed", gold=run.inspect_request().observation.resources[6])
                visited.add(target.handle)
                break
        else:
            expected = {"chest": "chest", "arena": "arena", "school_magic": "school_magic", "school_war": "school_war"}[fixture]
            base.ensure(family == expected, "Rewardable object produced the wrong delivered family")
            if fixture.startswith("school") and first:
                before, after, _, _ = run.answer("cancel")
                exact_resource_delta(before.observation.resources, after.observation.resources, 0)
                base.ensure(run.hero_window().primary_skills == prior.primary_skills, "School cancel changed primary skills")
                run.revisit(target)
                base.ensure(run.family() == expected, "Cancel incorrectly consumed the school's per-hero visit")
                seen.add("cancel")
            before, after, extra, choice = run.answer("select_component", selection % 2)
            component = before.observation.window.dialog_panel.components[choice.component_index]
            current = run.hero_window()
            gold = component.amount if component.kind == "resource" else -1000 if fixture.startswith("school") else 0
            exact_resource_delta(before.observation.resources, after.observation.resources, gold)
            if component.kind == "primary_skill":
                index = ("attack", "defence", "spellpower", "knowledge").index(component.skill)
                skills = list(prior.primary_skills); skills[index] += component.amount
                base.ensure(tuple(skills) == current.primary_skills, "Native primary-skill reward differs from its displayed choice")
            elif component.kind == "experience":
                base.ensure(current.level == 1, "Diagnostic chest changed the preserved level cap")
            seen.add(component.kind + ":" + str(component.skill if component.kind == "primary_skill" else choice.component_index))
            run.check("native_rewardable_selected_effect", "passed", family=family,
                      displayed_component=asdict(component), gold_delta=gold, level=current.level)
            if fixture != "chest" and first:
                repeat = run.revisit(target)
                expected_notice = {"arena": "core.advevent.1", "school_magic": "core.advevent.72",
                                   "school_war": "core.advevent.159"}[fixture]
                base.ensure(run.family() is None and expected_notice in notice_templates(repeat),
                            "Native revisited rewardable lacked its ordinary visited notice")
                seen.add("revisit")
            selection += 1
        first = False
        visited.add(target.handle)
    required = {"chest": {"resource:0", "experience:1", "core.advevent.145"},
                "arena": {"primary_skill:attack", "primary_skill:defence", "revisit"},
                "school_magic": {"primary_skill:spellpower", "primary_skill:knowledge", "cancel", "revisit", "core.advevent.73"},
                "school_war": {"primary_skill:attack", "primary_skill:defence", "cancel", "revisit", "core.advevent.160"},
                "scholar": {"scholar_primary_skill", "scholar_secondary_skill", "scholar_spell"}}[fixture]
    missing = sorted(required - seen)
    run.check("declared_rewardable_branch_coverage", "unverified" if missing else "passed",
              observed=sorted(seen), missing=missing, visited_objects=len(visited), native_rng_not_overridden=True)


def scholar_exchange_case(run):
    hero = run.own_hero()
    observation = run.inspect_request().observation
    partner = next(h for h in observation.heroes if h.handle != hero)
    target = next(obj for obj in observation.objects if obj.handle == partner.handle)
    additions = run.visit(target)
    base.ensure(run.inspect_request().observation.context.kind == "exchange", "Own Scholar meeting did not open a native exchange")
    notices = [frame.event.notice for frame in additions if frame.event is not None and frame.event.kind == "notice"
               and any(token.text_id == "core.genrltxt.139" for token in frame.event.notice.text)]
    base.ensure(len(notices) == 1 and notices[0].components[0].kind == "secondary_skill"
                and notices[0].components[0].content_id == "core:scholar", "Own Scholar meeting omitted its delivered skill notice")
    spells = [component.content_id for component in notices[0].components[1:]]
    base.ensure(set(spells) == {"core:magicArrow", "core:haste"}, "Native Thane/Josephine exchange did not show both starting spells")
    before, more, result = run.call(run.action("close"), "explicit_close_own_scholar_exchange")
    proof = native_receipt(before, more, result, "closed")
    run.check("native_own_hero_scholar_notice", "passed", notice=asdict(notices[0]),
              closed_exchange_receipt=proof, spellbook_state_not_exposed=True)


def fixture_identity(fixture, player, directory=MAPS):
    path = directory / (scenario_id(fixture, player) + ".vmap")
    verify_dialog_scenario(path, fixture, player)
    manifest_path = path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    base.ensure(manifest["map_sha256"] == digest(path) and manifest["training_input"] is False,
                "Diagnostic fixture manifest does not bind its exact map")
    return {"path": str(path.resolve()), "sha256": digest(path), "manifest": str(manifest_path.resolve()),
            "manifest_sha256": digest(manifest_path), "fixture": fixture, "player": player,
            "training_input": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "data", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--maps", type=Path, default=MAPS)
    parser.add_argument("--case", choices=(*CASES, "all"), default="all")
    parser.add_argument("--player", choices=(*base.COLORS, "both"), default="both")
    parser.add_argument("--seed", type=int, default=107001)
    parser.add_argument("--seed-count", type=int, default=1)
    parser.add_argument("--max-days", type=int, default=8)
    parser.add_argument("--max-decisions", type=int, default=192)
    parser.add_argument("--case-seconds", type=float, default=120)
    parser.add_argument("--watchdog", type=float, default=30)
    parser.add_argument("--startup-seconds", type=float, default=30)
    args = parser.parse_args(argv)
    if not 1 <= args.seed <= 2147480000 or not 1 <= args.seed_count <= 16 or not 1 <= args.max_days <= 12 or not 1 <= args.max_decisions <= 256 or any(
            not math.isfinite(value) or not 0 < value <= 180 for value in (args.case_seconds, args.watchdog, args.startup_seconds)):
        parser.error("Use bounded positive seeds, days, actions and wall limits")
    for name in ("binary", "data", "output", "maps"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    base.ensure(not args.output.exists(), "Preserve prior runs; output already exists")
    args.interface = INTERFACE
    cases = tuple(CASES) if args.case == "all" else (args.case,)
    colors = base.COLORS if args.player == "both" else (args.player,)
    fixtures = {(CASES[case][0], color): fixture_identity(CASES[case][0], color, args.maps) for case in cases for color in colors}
    schedule = [{"case": case, "player": color, "repeat": repeat,
                 "seed": args.seed + 64 * tuple(CASES).index(case) + 2 * repeat + base.COLORS.index(color)}
                for case in cases for repeat in range(args.seed_count) for color in colors]
    for identity in fixtures.values():
        environment.VcmiLauncher(args.binary, args.data, Path(identity["path"])).preflight()
    subprocess.run([sys.executable, str(ROOT / "tools/prepare_vcmi.py"), "--check"], cwd=ROOT, check=True, timeout=60)
    implementation = implementation_identity()
    for name in ("experiments/explorer/create_dialog_scenarios.py", "validation/probes/validate_adventure_dialogs.py"):
        implementation[name] = digest(ROOT / name)
    native = runtime_identity(args.binary.parent)
    args.output.mkdir(parents=True)
    plan = {"schema": "heroes-adventure-dialogs-diagnostic-plan-v1", "interface": INTERFACE,
            "schedule": schedule, "fixtures": list(fixtures.values()), "implementation": implementation,
            "native_runtime": native, "binary_sha256": digest(args.binary),
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "bounds": {key: getattr(args, key) for key in ("max_days", "max_decisions", "case_seconds", "watchdog", "startup_seconds")},
            "training_performed": False, "training_input": False, "optimizer_steps": 0,
            "missing_branches": "Unverified; every predeclared seed is retained, with no outcome injection or adaptive resampling"}
    write_json(args.output / "plan.json", plan)
    plan_hash = digest(args.output / "plan.json")
    report = {"schema": "heroes-adventure-dialogs-native-validation-v1", "plan_sha256": plan_hash,
              "status": "running", "cases": [], "not_started": len(schedule),
              "training_performed": False, "training_input": False, "optimizer_steps": 0}
    def check_identity():
        for name, value in implementation.items():
            base.ensure(digest(ROOT / name) == value, "Diagnostic source changed during capture")
        base.ensure(runtime_identity(args.binary.parent) == native and digest(args.binary) == plan["binary_sha256"]
                    and digest(args.output / "plan.json") == plan_hash, "Diagnostic runtime or plan changed")
        for (fixture, color), expected in fixtures.items():
            base.ensure(fixture_identity(fixture, color, args.maps) == expected, "Diagnostic map changed")
    for row in schedule:
        run = None
        attempt = {**row, "status": "running"}; report["cases"].append(attempt)
        write_json(args.output / "report.json", report)
        try:
            with wall_limit(args.case_seconds):
                check_identity()
                fixture, choice = CASES[row["case"]]
                args.player, args.map = row["player"], Path(fixtures[(fixture, row["player"])]["path"])
                run = DialogRun(f"{row['case']}-{row['player']}-{row['repeat']:02d}", args, environment, c, auditor, row["seed"])
                run.start()
                if fixture in ("guarded_mine", "abandoned_mine"): mine_case(run, choice)
                elif fixture.startswith("neutral_"): neutral_case(run, fixture, choice)
                elif fixture == "scholar_exchange": scholar_exchange_case(run)
                else: rewardable_case(run, fixture)
                run.finish(); run.persist_and_audit()
                base.ensure(run.end.status == "truncated", "Diagnostic did not finish at a paired unscored cutoff")
                summary = run.summary()
                attempt.update(status=summary["probe_status"], mechanics=summary, episode_directory=str(run.env.episode_dir))
                check_identity()
        except Exception as error:
            attempt.update(status="failed", error=f"{type(error).__name__}: {error}")
            if run is not None:
                retain_failed_evidence(run.env, None, attempt, error)
                if run.env.result is not None:
                    run.end = run.state = run.env.result
                    if not run.audit_attempted:
                        try:
                            with wall_limit(30): run.persist_and_audit()
                        except Exception as secondary: attempt["failure_history_audit_error"] = f"{type(secondary).__name__}: {secondary}"
                attempt["mechanics"] = run.summary(attempt["error"])
        finally:
            if run is not None:
                try: run.env.close()
                except Exception as error: attempt.update(status="failed", cleanup_error=f"{type(error).__name__}: {error}")
            try: check_identity()
            except Exception as error: attempt.update(status="failed", identity_error=f"{type(error).__name__}: {error}")
            report["not_started"] = len(schedule) - len(report["cases"])
            write_json(args.output / "report.json", report)
        if attempt["status"] == "failed": break
    report["status"] = "failed" if report["not_started"] or any(x["status"] == "failed" for x in report["cases"]) else (
        "unverified" if any(x["status"] == "unverified" for x in report["cases"]) else "passed")
    write_json(args.output / "report.json", report)
    print(json.dumps({"status": report["status"], "attempts": len(report["cases"]), "not_started": report["not_started"]}))
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
