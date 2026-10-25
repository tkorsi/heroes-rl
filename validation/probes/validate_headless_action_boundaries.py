#!/usr/bin/env python3
"""Controlled boundary probes; no engine process, real game, or training.

Run from any directory with --source pointing at the reviewed PR checkout and
--expected-head pinning its commit. Imports that checkout's production adapter
and existing tests/test_headless.py message builders, injects socket/process
failures in memory, and writes JSON to --output (boundary.json beside this script
by default). Nonzero exit means probe setup failed; finding statuses distinguish
bug reproduction from desired behavior passing. No learner import is needed.
"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import argparse
import hashlib
import importlib.util
import inspect
import json
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time

LOG = ("Experiment player 0 won. Terminal received.\n"
       "Experiment player 1 lost. Terminal received.\n"
       "Experiment self-play complete. Ending game.\n")


def load_reviewed_source(source):
    global env, history, observation, columns, obj, update, OWN
    fixture_path = source / "tests/test_headless.py"
    spec = importlib.util.spec_from_file_location("boundary_probe_fixtures", fixture_path)
    fixtures = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = fixtures
    spec.loader.exec_module(fixtures)
    env = fixtures.env_module
    history, observation, columns = fixtures.history, fixtures.observation, fixtures.columns
    obj, update, OWN = fixtures.obj, fixtures.update, fixtures.OWN


def function_ref(function):
    lines, first = inspect.getsourcelines(function)
    return [first, first + len(lines) - 1]


def action_consumption():
    player_history = history()
    hero = obj(1, 0, 0, owner=0)
    player_history.ingest(observation(0, columns(0, 2), [hero], [
        update(0, appeared=[hero], revealed=columns(0, 2), own=OWN)]))
    validator = env.BridgeValidator(0, "1")
    validator.ready = True
    validator.pending = ("action", 0, [True] * 9)
    side = SimpleNamespace(validator=validator, connection=Mock(), history=player_history,
                           actions_sent=0, decisions_sent=0, responses_undelivered=0, write_error=None)
    side.connection.sendall.side_effect = BrokenPipeError("engine closed after terminal")
    instance = env.HeadlessSelfPlayEnv.__new__(env.HeadlessSelfPlayEnv)
    instance.request = SimpleNamespace(player=0)
    instance.sides = {0: side}
    instance.request_time = time.monotonic()
    instance.agent_wait = 0
    instance.costs = {"respond": 0, "act": 0}
    instance.env_step = 0
    instance.step_latencies = []
    instance._write_ledger = lambda *args: None
    instance._advance = lambda: None
    before = len(player_history.v1.frames)
    instance.step(2)
    executions = {color: {"ready": True, "actions_sent": 1, "error": None,
                          "terminal": {"outcome": outcome},
                          "write_error": side.write_error if color == "red" else None}
                  for color, outcome in (("red", "win"), ("blue", "loss"))}
    classification = env.scoring.classify_headless_self_play(0, None, LOG, executions)
    added = player_history.v1.frames[before:]
    reproduced = side.actions_sent == 1 and any(f["kind"] == "action" for f in added)
    return {
        "finding": "P1 failed-transport action folding", "bug_reproduced": reproduced,
        "desired_behavior_passed": side.actions_sent == 0 and not any(f["kind"] == "action" for f in added),
        "setup": "Valid pending Red action; sendall raises BrokenPipeError. Actual step(), history fold and classifier execute. Only transport, next advance and unrelated ledger I/O are stubbed.",
        "observed": {"write_error": side.write_error, "actions_sent_delta": side.actions_sent,
                     "responses_undelivered": side.responses_undelivered,
                     "added_frames": added, "classification_with_write_error": classification},
        "expected": "A failed or unconsumed response must not become a taken-action sample or receive terminal credit. Preserve the preceding unacted decision, reconcile consumption, or exclude uncertain learner data. A separately verified natural outcome may remain reportable.",
        "source": {"experiments/headless/env.py": function_ref(env.HeadlessSelfPlayEnv.step),
                   "experiments/episodes/scoring.py": function_ref(env.scoring.classify_headless_self_play)},
    }


def finalization_order(order):
    with tempfile.TemporaryDirectory(prefix="heroes-boundary-recheck-") as directory:
        instance = env.HeadlessSelfPlayEnv(None, Path(directory), replay=False)
        instance.episode_dir = Path(directory)
        (instance.episode_dir / "logs").mkdir()
        instance.log = (instance.episode_dir / "logs/process.log").open("w")
        instance.log.write(LOG)
        instance.process = Mock()
        instance.process.wait.return_value = 0
        instance.started = time.monotonic()
        instance.reason = instance.detail = None
        instance.stopping = threading.Event()
        instance.shutdown_deadline = None
        instance.episode, instance.seed, instance.created_at = 1, 7, "controlled probe"
        instance.engine_wait = instance.agent_wait = 0
        instance.costs, instance.step_latencies, instance.launch_info = {}, [], {}
        instance.sides = {
            player: SimpleNamespace(color=color, reader=None, connection=None, ledger=None, cutoff=None,
                                    terminal_at=0, ready_at=0, decode_seconds=0,
                                    unread_from=0, unread_frames_from=0,
                                    validator=SimpleNamespace(ready=True, terminal=outcome),
                                    history=SimpleNamespace(records=[], v1=SimpleNamespace(frames=[])))
            for player, (color, outcome) in enumerate((("red", "win"), ("blue", "loss")))
        }
        instance._execution = lambda side: {
            "ready": True, "actions_sent": 1, "error": None,
            "terminal": {"outcome": "win" if side.color == "red" else "loss"}}
        instance._counters = lambda side: {}
        instance._identity = lambda: {}
        instance._engine_evidence = lambda text: ({}, None)
        events = {
            "exit": ("exit", None, 0, time.monotonic()),
            "bridge_error": ("bridge_error", 0,
                             "ProtocolError: Bridge stream ended inside a message", time.monotonic()),
        }
        for kind in order:
            instance.events.put(events[kind])
        # The process is a Mock; this patch guarantees the failure branch cannot
        # address a real process group. All queue handling and scoring are real.
        with patch.object(env, "stop_process_group", lambda process: None):
            result = instance._advance()
        return {"queued_order": order, "status": result.status,
                "termination_reason": result.termination_reason, "rewards": result.rewards,
                "failure_detail": result.manifest["failure_detail"]}


def native_watermark(source):
    """Exercise exact issuance invocation and method under both mutex orders."""
    cpp = (source / "engine/ExplorerRL/ExplorerRL.cpp").read_text()
    # Preserve offsets while hiding comments/literals from brace matching.
    masked = re.sub(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"', lambda match: " " * len(match[0]), cpp)
    def region(start, opening, closing):
        begin = masked.index(opening, start)
        depth = 1
        end = begin + 1
        while depth:
            depth += (masked[end] == opening) - (masked[end] == closing)
            end += 1
        return cpp[start:end]
    start = cpp.index("bool ExplorerRL::executeRequest(")
    execution = region(start, "{", "}")
    call_start = cpp.index("const bool accepted = executeRequest([this, action, heroID, position]", cpp.index("void ExplorerRL::runTurn("))
    invocation = region(call_start, "(", ")") + ";"
    fixture = r'''
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <future>
#include <iostream>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
enum class EPlayerStatus { INGAME, ENDED };
struct int3 { int x=0,y=0,z=0; int3()=default; int3(int a,int b,int c):x(a),y(b),z(c){} bool operator!=(const int3&)const{return false;} };
struct Hero { int getOwner()const{return 0;} int3 visitablePos()const{return {};} int3 convertFromVisitablePos(int3 p)const{return p;} };
class ExplorerRL;
struct Callback {
    ExplorerRL * owner;
    EPlayerStatus status = EPlayerStatus::INGAME;
    EPlayerStatus getPlayerStatus(int, bool) const { return status; }
    void endTurn();
    Hero * getHero(int) { static Hero hero; return &hero; }
    void moveHero(const Hero *, int3, bool) { endTurn(); }
};
class ExplorerRL {
public:
    std::mutex stateMutex;
    std::shared_mutex gameMutex;
    std::condition_variable changed;
    int pendingRequest = -1, actionID = 0, submittedActionID = -1, timeoutSeconds = 1, playerID = 0, requests = 0, selectedAction = 8;
    bool terminal = false, stopping = false, moving = false, inBattle = false;
    bool observedIssued = false, observedAccepted = false;
    std::string failure;
    std::map<int, bool> acknowledgements;
    std::deque<int> decisions;
    Callback actualCallback{this}; Callback * callback = &actualCallback;
    std::promise<void> * readAttempted = nullptr, * requestEntered = nullptr;
    std::shared_future<void> releaseRequest;
    static constexpr int DIRECTIONS[8][2] = {};
    std::shared_lock<std::shared_mutex> readGameState() {
        if(readAttempted) readAttempted->set_value();
        return std::shared_lock(gameMutex);
    }
    void handleDecision(int) {}
    bool executeRequest(const std::function<void()> &, bool, int64_t, bool *);
    void issue() {
        const int action = selectedAction, heroID = 0; const int3 position;
        bool issued = false;
        INVOCATION
        observedIssued = issued; observedAccepted = accepted;
    }
    int terminalSnapshot() {
        std::unique_lock game(gameMutex);
        std::lock_guard state(stateMutex);
        actualCallback.status = EPlayerStatus::ENDED;
        terminal = true;
        return submittedActionID;
    }
};
void Callback::endTurn() {
    {
        std::lock_guard state(owner->stateMutex);
        ++owner->requests; owner->pendingRequest = 42; owner->acknowledgements[42] = true;
    }
    if(owner->requestEntered) owner->requestEntered->set_value();
    if(owner->releaseRequest.valid()) owner->releaseRequest.wait();
}
EXECUTION
int main() {
    ExplorerRL terminalFirst;
    std::promise<void> attempted; terminalFirst.readAttempted = &attempted;
    std::unique_lock held(terminalFirst.gameMutex);
    std::thread first([&]{ terminalFirst.issue(); });
    attempted.get_future().wait();
    terminalFirst.actualCallback.status = EPlayerStatus::ENDED; terminalFirst.terminal = true;
    const int beforeWatermark = terminalFirst.submittedActionID;
    held.unlock(); first.join();

    ExplorerRL issuanceFirst;
    std::promise<void> entered, release, terminalAttempted;
    issuanceFirst.requestEntered = &entered; issuanceFirst.releaseRequest = release.get_future().share();
    std::thread second([&]{ issuanceFirst.issue(); });
    entered.get_future().wait();
    auto terminalResult = std::async(std::launch::async, [&]{ terminalAttempted.set_value(); return issuanceFirst.terminalSnapshot(); });
    terminalAttempted.get_future().wait();
    const bool terminalBlocked = terminalResult.wait_for(std::chrono::seconds(0)) != std::future_status::ready;
    release.set_value(); second.join();
    const int afterWatermark = terminalResult.get();

    ExplorerRL endedBeforeCallback;
    endedBeforeCallback.actualCallback.status = EPlayerStatus::ENDED;
    endedBeforeCallback.issue();
    std::cout << "{\"terminal_first\":{\"watermark\":" << beforeWatermark << ",\"requests\":" << terminalFirst.requests
              << ",\"issued\":" << (terminalFirst.observedIssued?"true":"false")
              << "},\"issuance_first\":{\"watermark\":" << afterWatermark << ",\"requests\":" << issuanceFirst.requests
              << ",\"issued\":" << (issuanceFirst.observedIssued?"true":"false")
              << ",\"terminal_blocked_during_request\":" << (terminalBlocked?"true":"false")
              << "},\"status_ended_before_gameover_callback\":{\"watermark\":" << endedBeforeCallback.submittedActionID
              << ",\"requests\":" << endedBeforeCallback.requests
              << ",\"issued\":" << (endedBeforeCallback.observedIssued?"true":"false") << "}}\n";
}
'''.replace("INVOCATION", invocation).replace("EXECUTION", execution)
    compiler = shutil.which("clang++") or shutil.which("c++")
    if compiler is None:
        raise RuntimeError("A C++ compiler is required for the extracted-code boundary probe")
    with tempfile.TemporaryDirectory(prefix="heroes-watermark-probe-") as directory:
        path = Path(directory)
        (path / "probe.cpp").write_text(fixture)
        compiled = subprocess.run([compiler, "-std=c++20", "-pthread", str(path / "probe.cpp"), "-o", str(path / "probe")],
                                  capture_output=True, text=True, timeout=30)
        if compiled.returncode:
            raise RuntimeError(compiled.stderr)
        result = subprocess.run([str(path / "probe")], capture_output=True, text=True, check=True, timeout=5)
    observed = json.loads(result.stdout)
    passed = (observed["terminal_first"] == {"watermark": -1, "requests": 0, "issued": False}
              and observed["issuance_first"] == {"watermark": 0, "requests": 1, "issued": True, "terminal_blocked_during_request": True}
              and observed["status_ended_before_gameover_callback"] == {"watermark": -1, "requests": 0, "issued": False})
    return {
        "finding": "P1 native issuance and terminal ordering",
        "bug_reproduced": not passed,
        "desired_behavior_passed": passed,
        "observed": observed,
        "expected": "When a terminal wins before the request is issued, its watermark must stay at the previous action (-1 here) and the late request must not issue. Commit consumption within the same game-state-locked request-issuance boundary as the terminal exclusion check.",
        "source": {"engine/ExplorerRL/ExplorerRL.cpp": {"method_start": cpp[:start].count("\n") + 1,
                                                          "invocation_start": cpp[:call_start].count("\n") + 1}},
        "production_extract_sha256": hashlib.sha256((invocation + execution).encode()).hexdigest(),
        "scope": "Exact production runTurn issuance invocation/lambda and executeRequest method; surrounding state, callback and game mutex are fixture supplies. Deterministic barriers test terminal-first, issuance-first, and ended player status before gameOver callback. No game or engine build.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Reviewed PR checkout root")
    parser.add_argument("--expected-head", required=True, help="Exact expected Git commit")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("boundary.json"))
    parser.add_argument("--tests-log", type=Path, help="Existing targeted unittest output; read only")
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if head != args.expected_head:
        raise SystemExit(f"Expected exact reviewed head {args.expected_head}, found {head}")
    load_reviewed_source(source)
    action = action_consumption()
    native = native_watermark(source)
    exit_first = finalization_order(["exit", "bridge_error"])
    error_first = finalization_order(["bridge_error", "exit"])
    stream = {
        "finding": "P2 stream finalization",
        "bug_reproduced": exit_first["status"] == "completed" and error_first["status"] == "failed",
        "desired_behavior_passed": all(result["status"] == "failed" and all(v is None for v in result["rewards"].values())
                                       for result in (exit_first, error_first)),
        "setup": "Actual _advance/_finish/scoring with identical terminal/log evidence and queued exit plus reader error; only event ordering changes. Process, unrelated reporting, and metadata are stubbed. No process is launched.",
        "observed": {"exit_first": exit_first, "reader_error_first": error_first},
        "expected": "Both orders fail closed with null rewards; a queued reader protocol error must survive final draining.",
        "source": {"experiments/headless/env.py": [function_ref(env.HeadlessSelfPlayEnv._advance),
                                                     function_ref(env.HeadlessSelfPlayEnv._finish)]},
    }
    sources = ["experiments/headless/env.py", "experiments/episodes/scoring.py", "engine/ExplorerRL/ExplorerRL.cpp"]
    output = {"schema_version": 1, "reviewed_commit": head,
              "command": shlex.join([sys.executable, *sys.argv]),
              "source": str(source),
              "scope": "Controlled Python injection and extracted native-code compilation only; no project source/test edits, engine build, real games, benchmark interference or training.",
              "source_sha256": {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in sources},
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "findings": [action, stream, native]}
    if args.tests_log is not None:
        tests_log = args.tests_log.expanduser().resolve()
        log_text = tests_log.read_text()
        match = re.search(r"Ran (\d+) tests in ([\d.]+)s", log_text)
        output["unit_tests"] = {"log": str(tests_log),
                                "sha256": hashlib.sha256(tests_log.read_bytes()).hexdigest(),
                                "tests": int(match[1]) if match else None,
                                "seconds": float(match[2]) if match else None,
                                "passed": bool(match and log_text.rstrip().endswith("OK")),
                                "scope": "tests.test_headless and tests.test_headless_runner with the current learner contract supplied via HEROES_LEARNER_PATH; synthetic fixture, no VCMI game"}
    destination = args.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"reviewed_commit": head, "output": str(destination),
                      "findings": [{"finding": f["finding"], "bug_reproduced": f["bug_reproduced"],
                                    "desired_behavior_passed": f["desired_behavior_passed"]}
                                   for f in output["findings"]]}, indent=2))


if __name__ == "__main__":
    main()
