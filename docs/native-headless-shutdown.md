# Ordinary headless AI shutdown order

On the tested Linux/oneTBB runtime, stopping and joining the client network
thread before draining Nullkiller tasks can crash after an otherwise valid
natural terminal barrier. GDB identifies the worker in `yourTurn` task-handle
completion releasing a thread-reference wait vertex after its submitting thread
has gone; main is in `AIGateway::finish`.

The headless-only repair in `CServerHandler::endNetwork` first closes packet
callback admission under the existing interface mutex, then drains the client AI
tasks while the network thread remains alive, and only then stops/joins that
thread. Existing later gameplay cleanup is retained and its finish calls are
idempotent. The graphical path is unchanged. This does not enable EngineSession
or any reusable/fork worker and changes no rule, RNG, reward or policy behavior.

Build/check the canonical recipe and run the dedicated regression:

```sh
python3 tools/prepare_vcmi.py --check
cmake --build vcmi/build --target vcmiclient vcmireplay native_headless_shutdown_order --parallel 2
python3 -m unittest tests.test_native_headless_shutdown_order \
  tests.test_native_terminal_cancellation tests.test_native_shutdown \
  tests.test_native_request_lock_order tests.test_native_terminal_barrier \
  tests.test_native_terminal_accounting -v
```

The native fixture uses actual `CServerHandler::endNetwork`, an actual network
thread and actual `AsyncRunner`/TBB work. It fails against the old function and
passes 100 iterations with the repair. It contains no game rules or asset
substitutes. Two separate fresh ordinary end-turn-only games, covering both
learner seats, finish with exit zero and pass the unchanged contextual native
verifier, including full private-view messages and terminal accounting. They do
not replace Teacher's failed 72-choice gate. Rebuild and rerun that restored
policy's fresh Linux gate before training; preserve the original failure
unscored. Compact evidence: `validation/native-headless-shutdown-2026-10-25.json`.

## Separate terminal-attempt contract extension

A later fresh Linux workload exposed a different case: one native
`TradeOnMarketplace` request issued while the final authoritative terminal
packets were being consumed never reached server application. Closing admission
after the first client terminal callback cannot prevent requests already issued
before it; that attempted guard was not retained.

The existing request-accounting contract now recognizes this fourth concrete
packet type alongside `MoveHero`, `BuildStructure` and `EndTurn`. Its proof
requirements are unchanged: the independently replayed final terminal input and
its exact packet range, one opposite-seat native request with the immediately
next ID, decoded bytes bound to metadata, the matching actual terminal close,
no later learner response, and complete agreement of all applied inputs,
packets, states and client history. Unknown request types remain rejected. The
attempt is counted and retained as **unapplied**, never executed, charged or
invented as a successful trade. No game rule, opponent algorithm or learner
behavior changes; this is an explicit verifier-contract extension.

The real packet/codec tamper matrix covers both seats and all four admitted
types (460 checks). Retained failed captures may have separately named
supplemental diagnostics under the extended contract; the original failed rows
remain unscored. Fresh games and their unchanged native checks establish new
admission. Teacher owns any exact checkpoint/runtime-binding transition and
must rerun its current gate before training.
