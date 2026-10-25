"""Executable pinned-source/config audit, not timer or initial-terminal gameplay.

These assertions deliberately fail when the audited headless profile or native
gates change. They prove that this profile excludes timers and that unsupported
origins cannot receive a receipt; they do not claim coverage of other profiles.
"""
import ast
import inspect
import json
from pathlib import Path
import re
import tempfile
import textwrap
import unittest

from experiments.headless import contextual_evidence
from experiments.headless.env import VcmiLauncher

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "vcmi"


def body(source, signature):
    """Extract the source body of the exact audited native function."""
    start = source.index("{", source.index(signature))
    depth = 1
    cursor = start + 1
    while depth:
        if source[cursor] == "{": depth += 1
        elif source[cursor] == "}": depth -= 1
        cursor += 1
    return source[start + 1:cursor - 1]


class TerminalProfileConfigTests(unittest.TestCase):
    def test_isolated_launcher_uses_headless_testmap_without_timer_settings(self):
        settings = VcmiLauncher(Path("unused-client"), Path("unused-data"), Path("unused.vmap")).settings(123)
        def keys(value):
            if type(value) is dict:
                for key, child in value.items():
                    yield key
                    yield from keys(child)
            elif type(value) is list:
                for child in value: yield from keys(child)
        self.assertFalse(any("timer" in key.lower() for key in keys(settings)))
        self.assertFalse(settings["server"]["useProcess"])
        tree = ast.parse(textwrap.dedent(inspect.getsource(VcmiLauncher.launch)))
        commands = [node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "command" for target in node.targets)]
        self.assertEqual(len(commands), 1)
        constants = [node.value for node in ast.walk(commands[0]) if isinstance(node, ast.Constant)]
        self.assertIn("--headless", constants)
        self.assertIn("--testmap", constants)
        self.assertNotIn("--testsave", constants)
        self.assertFalse(any("timer" in item.lower() for item in constants if isinstance(item, str)))
        # Source/config evidence only. The actual launched StartInfo is bound by
        # the replay init record in each separately validated native episode.

    def test_matching_initial_or_tick_receipts_cannot_be_accepted(self):
        for origin in ("start", "tick"):
            with self.subTest(origin=origin), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "engine").mkdir()
                common = {"schema": contextual_evidence.SCHEMA, "episode": "1", "interface": "adventure-windows-v1",
                          "barrier_id": 1, "origin": origin, "request": None, "client_packets": 1,
                          "outcomes": ["win", "loss"]}
                for side, path in zip(("server", "client"), contextual_evidence.FILES):
                    record = {**common, "kind": side}
                    if side == "client": record.update(callbacks_complete=True, observed_count=1)
                    (root / path).write_text(json.dumps(record) + "\n")
                with self.assertRaisesRegex(ValueError, "identity or outcomes mismatch"):
                    contextual_evidence.terminal_barrier(root, episode="1", interface="adventure-windows-v1",
                                                        outcomes=["win", "loss"])


@unittest.skipUnless((ENGINE / "server/TurnTimerHandler.cpp").is_file(), "Requires prepared pinned VCMI source for source audit")
class TerminalProfileNativeSourceTests(unittest.TestCase):
    def test_pinned_timer_defaults_and_headless_ai_release(self):
        pin = json.loads((ROOT / "dependencies/vcmi.json").read_text())
        self.assertEqual(pin["commit"], "6901839ae9760dfcd1d5ff2c4aa4ef6968fee752")
        timer = (ENGINE / "lib/TurnTimerInfo.h").read_text()
        defaults = dict(re.findall(r"\bint\s+(turnTimer|baseTimer|battleTimer|unitTimer)\s*=\s*(\d+)\s*;", timer))
        self.assertEqual(defaults, {name: "0" for name in ("turnTimer", "baseTimer", "battleTimer", "unitTimer")})
        start_info = (ENGINE / "lib/StartInfo.h").read_text()
        self.assertIn("TurnTimerInfo turnTimerInfo;", start_info)
        entry = (ENGINE / "clientapp/EntryPoint.cpp").read_text()
        for flag in ("headless", "testmap"):
            selected = body(entry, f'if(vm.count("{flag}"))')
            self.assertIn('session["onlyai"].Bool() = true;', selected)
        server = (ENGINE / "client/CServerHandler.cpp").read_text()
        start = body(server, "void CServerHandler::debugStartTest(")
        release = start.index("setPlayer(myFirstColor());")
        released = start.index("myFirstColor() == PlayerColor::CANNOT_DETERMINE", release)
        begin = start.index("sendStartGame();", released)
        self.assertLess(release, released)
        self.assertLess(released, begin)
        self.assertNotIn("setTurnTimerInfo", start)

    def test_timer_effects_require_human_players_and_enabled_timers(self):
        source = (ENGINE / "server/TurnTimerHandler.cpp").read_text()
        update = body(source, "void TurnTimerHandler::update(")
        self.assertRegex(update, r"if\(!gameHandler\.gameInfo\(\)\.getStartInfo\(\)->turnTimerInfo\.isEnabled\(\)\)\s*return;")
        turn = body(source, "void TurnTimerHandler::onPlayerMakingTurn(")
        human = body(turn, "if(state && state->human && timer.isActive && !timer.isBattle && state->status == EPlayerStatus::INGAME)")
        self.assertIn("onPlayerEndsTurn(state->color)", human)
        self.assertNotIn("onPlayerEndsTurn", turn[:turn.index("state && state->human")])
        battle = body(source, "void TurnTimerHandler::onBattleLoop(")
        self.assertRegex(battle, r"if\(!state \|\| state->status != EPlayerStatus::INGAME \|\| !state->human\)\s*return;")
        replay = (ROOT / "engine/ExperimentReplay/ReplayMain.cpp").read_text()
        self.assertRegex(replay, r'if\(init\["turn_timers"\]\.Bool\(\)\)\s*throw std::runtime_error\("Replays require games without turn timers"\);')

    def test_initial_and_tick_natural_endings_fail_closed_before_success(self):
        source = (ENGINE / "server/CGameHandler.cpp").read_text()
        reject = body(source, "void CGameHandler::rejectUnsupportedExperimentTerminalOrigin(")
        ended = body(reject, "if(contextualNaturalEnd())")
        self.assertIn('reportExperimentFailure(std::string("Unsupported contextual terminal origin: ") + origin)', ended)
        self.assertIn('throw std::runtime_error("Contextual natural completion outside a replayed request")', ended)
        self.assertNotIn("writeExperimentTerminalBarrier", ended)
        start = body(source, "void CGameHandler::start(")
        self.assertLess(start.index("turnOrder->onGameStarted();"), start.index('rejectUnsupportedExperimentTerminalOrigin("start");'))
        self.assertLess(start.index('rejectUnsupportedExperimentTerminalOrigin("start");'), start.index("replayRecorder->afterStart"))
        tick = body(source, "void CGameHandler::tick(")
        self.assertRegex(tick, r"if\(terminalBarrierSent\)\s*return;")
        self.assertLess(tick.index("turnTimerHandler->update"), tick.index('rejectUnsupportedExperimentTerminalOrigin("tick");'))


if __name__ == "__main__":
    unittest.main()
