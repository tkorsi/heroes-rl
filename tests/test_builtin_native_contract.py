"""Pure controller parser and pinned native source guards; no game or replay.

These checks establish narrow routing/evidence boundaries, not successful
production compilation or real-game mixed-opponent behavior.
"""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "vcmi"


def function(source, signature):
    start = source.index(signature)
    brace = source.index("{", start)
    depth, cursor = 1, brace + 1
    while depth:
        if source[cursor] == "{": depth += 1
        elif source[cursor] == "}": depth -= 1
        cursor += 1
    return source[start:cursor]


@unittest.skipUnless((ENGINE / "lib/callback/CDynLibHandler.cpp").is_file(), "Prepared pinned source required")
class BuiltinControllerParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("c++")
        if compiler is None:
            raise unittest.SkipTest("A C++20 compiler is required for the pure parser test")
        cls.directory = tempfile.TemporaryDirectory(prefix="heroes-builtin-parser-")
        cls.root = Path(cls.directory.name)
        source = (ENGINE / "lib/callback/CDynLibHandler.cpp").read_text()
        # Compile the actual root-patched parser, with no VCMI runtime/library.
        native = "\n".join(function(source, name) for name in
                             ("bool contextualEvidenceEnabled()", "int experimentBuiltinLearner()"))
        program = "#include <cstdlib>\n#include <iostream>\n#include <stdexcept>\n#include <string>\n" + native + "\n" + """
int main() {
    try { std::cout << experimentBuiltinLearner(); return 0; }
    catch(const std::exception &) { return 3; }
}
"""
        path = cls.root / "parser.cpp"
        path.write_text(program)
        cls.binary = cls.root / "parser"
        subprocess.run([compiler, "-std=c++20", str(path), "-o", str(cls.binary)], check=True,
                       capture_output=True, timeout=30)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def parse(self, overrides):
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("EXPLORER_RL_", "VCMI_EXPERIMENT_"))}
        env.update({k: v for k, v in overrides.items() if v is not None})
        return subprocess.run([str(self.binary)], env=env, capture_output=True, text=True, timeout=3)

    def test_absent_mixed_flag_preserves_legacy_mode(self):
        result = self.parse({"EXPLORER_RL_PLAYERS": "0,1"})
        self.assertEqual((result.returncode, result.stdout), (0, "-1"))

    def test_only_two_explicit_single_learner_mappings_are_accepted(self):
        base = {"EXPLORER_RL_BUILTIN_OPPONENT": "Nullkiller", "EXPLORER_RL_INTERFACE": "adventure-dialogs-v1",
                "VCMI_EXPERIMENT_CONTEXTUAL_EVIDENCE": "1", "EXPLORER_RL_PLAYERS": "0"}
        for seat in ("0", "1"):
            with self.subTest(seat=seat):
                result = self.parse({**base, "EXPLORER_RL_PLAYERS": seat})
                self.assertEqual((result.returncode, result.stdout), (0, seat))
        invalid = {"EXPLORER_RL_BUILTIN_OPPONENT": ("", "Nullkiller2", "ExplorerRL"),
                   "EXPLORER_RL_INTERFACE": (None, "movement-v1", "adventure-army-management-v1"),
                   "VCMI_EXPERIMENT_CONTEXTUAL_EVIDENCE": (None, "0", "true"),
                   "EXPLORER_RL_PLAYERS": (None, "", "0,1", "1,0", "2", "00", " 0", "0 "),
                   "EXPLORER_RL_PLAYER": ("0", "1")}
        for key, values in invalid.items():
            for value in values:
                with self.subTest(key=key, value=value):
                    self.assertEqual(self.parse({**base, key: value}).returncode, 3)


@unittest.skipUnless((ENGINE / "client/Client.cpp").is_file(), "Prepared pinned source required")
class BuiltinNativeSourceTests(unittest.TestCase):
    def test_startup_records_before_first_native_request_and_pins_real_rules(self):
        source = (ENGINE / "client/Client.cpp").read_text()
        init = function(source, "void CClient::initPlayerInterfaces()")
        self.assertIn("interfaceOrder = {PlayerColor(builtinLearner), PlayerColor(1 - builtinLearner)}", init)
        self.assertLess(init.index("installNewPlayerInterface("), init.index("sendRequest(air"))
        for condition in ("players.size() != 2", "difficulty != 0", "extraOptionsInfo.cheatsAllowed",
                          "isControlledByAI()", "getPlayerTeam(PlayerColor(0)) == gameState().getPlayerTeam(PlayerColor(1))"):
            self.assertIn(condition, init)
        self.assertIn(': builtinLearner >= 0 ? "Nullkiller"', init)

    def test_natural_barrier_requires_both_real_callbacks_and_own_delivery(self):
        source = (ENGINE / "client/NetPacksClient.cpp").read_text()
        visit = function(source, "void ApplyClientNetPackVisitor::visitPlayerEndsGame")
        self.assertLess(visit.index("callAllInterfaces(cl, &IGameEventsReceiver::gameOver"),
                        visit.index("recordBuiltinNativeTerminal("))
        source = (ENGINE / "client/Client.cpp").read_text()
        receipt = function(source, "bool CClient::recordBuiltinNativeTerminal")
        self.assertIn("!builtinNativeTerminalOutcomes.emplace(player, won).second", receipt)
        self.assertIn("if(player == PlayerColor(learner))", receipt)
        barrier = function(source, "void CClient::completeExperimentTerminalBarrier")
        for guard in ("builtinNativeTerminalOutcomes.size() == 2", "explorerTerminalsReceived == own",
                      "builtinNativeTerminalOutcomes.at(PlayerColor(0)) == barrier.redWon",
                      "builtinNativeTerminalOutcomes.at(PlayerColor(1)) != barrier.redWon",
                      "position.index != position.completed", "*lastExperimentAcknowledgement"):
            self.assertIn(guard, barrier)

    def test_opponent_replay_is_explicit_external_input_and_automatic_is_not_widened(self):
        source = (ROOT / "engine/ExperimentReplay/ContextualReplay.cpp").read_text()
        allowed = function(source, "bool automaticAllowed(")
        self.assertIn("dynamic_cast<const AdvInterfaceReady *>", allowed)
        self.assertIn("dynamic_cast<const MakeAction *>", allowed)
        self.assertNotIn("learner", allowed)
        self.assertNotIn("MoveHero", allowed)
        run = function(source, "JsonNode ContextualReplay::run(")
        accounting = (ROOT / "engine/ExperimentReplay/RequestAccounting.h").read_text()
        for guard in ('source == "native_opponent"', "request->player != PlayerColor(1 - learner)",
                      'transaction->first != "client"', "requestIndex != inputs.size()",
                      'item["pack"].String() != expected["pack"].String()',
                      'expected["request"].Integer() != request->requestID'):
            self.assertIn(guard, accounting)
        self.assertIn("return std::make_shared<PassiveOpponent>()", run)
        passive = source[source.index("class PassiveOpponent"):source.index("/// Automatic requests")]
        self.assertNotIn("selectionMade", passive)
        self.assertNotIn("endTurn", passive)
        self.assertNotIn("getNewAI", source)

    def test_mode_specific_schema_coverage_and_effective_fog_receipts(self):
        capture = (ROOT / "engine/ExplorerRL/ContextualEvidence.cpp").read_text()
        self.assertIn('builtin ? BUILTIN_SCHEMA : SCHEMA', capture)
        self.assertIn('loaded["native_opponent_requests"].Integer() = checked["native_opponent_requests"].Integer() = 0', capture)
        self.assertIn('if(remaining != 0) throw', capture)
        self.assertIn('result["controllers"] = metadata["controllers"]', capture)
        source = (ENGINE / "AI/Nullkiller/Engine/Nullkiller.cpp").read_text()
        self.assertIn('std::getenv("VCMI_EXPERIMENT_DIR")', source)
        self.assertIn('if(!cb->getStartInfo()->extraOptionsInfo.cheatsAllowed)', source)
        self.assertIn('useObjectGraph = false;', source)
        self.assertIn('openMap = false;', source)
        self.assertIn('Experiment AI visibility: player=%d cheats_allowed=%d open_map=%d use_object_graph=%d', source)


if __name__ == "__main__":
    unittest.main()
