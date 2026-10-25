# Embedded in-process engine (opt-in)

The embedded backend runs the pinned VCMI game inside the Python host process.
The path is Python/PyTorch, then `heroes_embedded_engine` (a thin CPython
binding), then the existing `EngineSession`, then the unchanged VCMI game state,
server handler, routing, BattleAI automatic battles and ExplorerRL. Nullkiller on
Easy is an explicitly owned in-process opponent object. No `vcmiclient` or
`vcmiserver` process, socket, line framing, reader thread or disk polling is on
the decision path.

It is opt-in and experimental. The cold executable backend stays the default
and the fallback. No learner source, model, reward, map or checkpoint changed,
and no training was run. Measurements and real-game acceptance are recorded in
[embedded engine validation](../validation/embedded-engine-2026-10-25.json).

## What runs where

| Concern | Cold backend | Embedded backend |
| --- | --- | --- |
| Game rules, state, RNG, routing, queries, battles | `vcmiclient` process with its in-thread server | Same VCMI code in the Python process (`EngineSession`, in-thread server, internal connection) |
| Player-visible stream | ExplorerRL, JsonWriter line over TCP, Python reader thread, `json.loads` | Same ExplorerRL records, handed over in memory and converted directly to Python values |
| Learner response | `json.dumps` line, VCMI strict JSON parser | Same line and the same parser (`submit_text`); 64 KiB admission kept |
| Opponent | Nullkiller Easy created by the client's AI loader | `NullkillerOpponent`: the same module, created per game through an installed provider, released and reset after the game |
| Automatic battles | BattleAI for both sides | Unchanged |
| Episode boundaries | Process exit status or SIGTERM/SIGKILL | `finish()` report: exit status 0 after the terminal barrier, or a clean host stop |
| Evidence | Engine capture files, bridge ledger, history, manifest | Same files and schemas, plus the close report; replayed by the unchanged `vcmireplay` |

The only transport change is the injected `InProcessPolicy` channel in
`engine/NativeSession/PolicyTransport.*`. It is compiled into `libvcmi`
next to the socket transport, which the cold client keeps using.

## Native API

`EngineSession` (C++20, `engine/NativeSession/EngineSession.h`):

- `load()` loads the filesystem, mods and game library once per process.
- `reset(EpisodeConfig)` creates a new game: settings baseline restored,
  per-game environment, seed, learner seat, capture directory and engine log,
  then a fresh `GameInstance`, server, game state, ExplorerRL and Nullkiller
  instance through the existing headless start path.
- `take(deadline, untilBoundary)` delivers every new player message once, plus
  a boundary token when a decision is newly offered, the terminal latch and the
  first failure. A high-water mark hands over a long opponent turn before the
  queue (1024 messages) could overflow; an overflow is a failure.
- `submit(token, response)` answers exactly the offered decision; stale or
  duplicate tokens are refused.
- `finish(natural, deadline)` waits for the natural terminal barrier when asked,
  then always joins every game thread, destroys the game, releases the opponent
  and returns an `EpisodeReport`.
- `observe`/`step`, `waitForShutdown` and `closeEpisode` remain for the existing
  C++ benchmark and contract tests.
- `poisoned()` and `active()`; `timings()` splits load, reset, advance, cleanup,
  natural finish wait, opponent turns and automatic battles.

The Python module (`PROTOCOL = "heroes-embedded-engine-v1"`) exposes
`Engine(console_log=False)` with `load`, `reset(episode=, map=, capture_directory=,
seed=, learner=, policy_timeout=30, log_path="")`, `take(timeout,
until_boundary=True)`, `submit(generation, decision, dict)`, `submit_text(generation,
decision, str)`, `finish(natural=, timeout=)`, `timings`, `close`, and the
`poisoned`/`active` properties. Module functions are `resources()` (RSS, peak RSS,
threads, descriptors) and test or measurement fixtures. The GIL is released
while the engine works. One `Engine` may exist per process.

`load/reset/take/submit/finish/close` is the justified equivalent of
load/reset/observe/step/close. `take` separates delivery from decision, so a
long opponent turn streams incrementally and a terminal or failure is never
hidden behind a pending step.

## Lifecycle, isolation and failure

Loaded once per process: VCMI filesystem, mod and game library resources, and
the dynamic AI libraries. Created and destroyed per game: the game instance,
server and handler, game state and its RNG seed, fog of war, queries and pending
dialogs, request ids, ExplorerRL and its contextual evidence, the Nullkiller
gateway and its worker tasks, the policy channel and the per-game log.

`NullkillerOpponent` has an explicit lifecycle. `initialize()` runs once after
load. `attach(player)` runs before each game and installs the provider, a
passive packet observer used only for timing, and the unchanged Easy and BattleAI
selections. The engine's own turn order advances it; the host never drives or
inspects it. `detach()` and `release()` run after all game threads joined.
Patch 0029 then returns Nullkiller's module-level caches (pathfinding node
storage, cached hero graphs) to fresh-process state. `shutdown()` runs with the
session. An instance that outlives its game poisons the session.

Any native failure, failed join, channel failure or experiment error poisons the
engine: the next `reset` is refused and the worker must start a new process.
The reviewed shutdown and poisoning fixes are retained. The embedded engine owns
the process working directory (VCMI's development layout resolves data and
libraries against `.`) and its profile environment (`XDG_*` on Linux,
`VCMI_USER_DATA_DIR` on macOS) for its lifetime.

## Learner contract

The learner sees exactly what the cold backend delivers: the same ExplorerRL
player-filtered records, in the same order, decoded to the same Python values.
Tests compare the direct conversion with the cold codec (JsonWriter text with
newlines removed, then `json.loads`) for floats, escaping, control and invalid
UTF-8 bytes. A message the cold reader would reject is reported as the same
protocol failure after the same prefix. Responses use the cold JSON line and
VCMI's strict parser, so admission and refusals are identical.

`EmbeddedBuiltinEnv` (`experiments/headless/embedded_env.py`) is a
`BuiltinHeadlessEnv` with the same reset, step, truncate, dialog cutoff, history,
manifest and finalization. The existing `BuiltinCollector` runs on it unchanged.
Hidden information, window-gated content, typed contextual actions including
`move_to(x, y)`, engine-built routes and natural, cutoff and failure
distinctions are those of the shared ExplorerRL and environment code.

## Evidence, identity and replay

An embedded manifest names `identity.backend`: the backend kind, protocol,
module path and SHA-256, and the SHA-256 of every `engine/NativeSession` source.
`ENGINE_RECIPE` and the learner checkpoint source identity are unchanged.
Replay verification (`experiments/headless/builtin_replay.py`) requires the
captured module and host sources to equal the current ones, then replays the
engine capture with the unchanged `vcmireplay` and checks the learner history.
Finalization (`experiments/headless/embedded_evidence.py`) proves:

- a natural ending by exit status 0 after the terminal barrier and a clean close;
- a host stop by the absence of any exit status, a clean join, a released
  opponent and no failure.

The cold rule (a SIGTERM/SIGKILL status) applies to cold manifests only.

## Build

Configure with the learner virtualenv's interpreter, so the module ABI matches:

```sh
cmake -S vcmi -B vcmi/build <usual options> -DENABLE_EMBEDDED_ENGINE=ON \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build vcmi/build --target vcmiclient vcmiserver ExplorerRL Nullkiller \
  BattleAI vcmireplay heroes_embedded_engine
```

macOS uses the README Homebrew options; Linux uses the
[cloud setup](cloud-environment.md) options. The module is written to
`vcmi/build/embedded/heroes_embedded_engine<EXT_SUFFIX>`, outside `bin/`, so the
executable runtime inventory is unchanged; its RPATH points at `build/bin`.
With the option on, the client and server static libraries are built
position-independent; game rules are unchanged.

Platform-specific code is limited to host services: resource telemetry
(`/proc` on Linux, Mach on macOS), the profile environment variable and the
extension suffix. No game-rule code differs by platform. As with the cold
backend, AI thread scheduling means a seed alone does not promise identical
autonomous games across runs, compilers or systems.

## Use

```python
from experiments.headless.embedded_env import EmbeddedLauncher, EmbeddedBuiltinEnv
launcher = EmbeddedLauncher(binary, data, game_map)          # build/bin/vcmiclient path, Data parent, map
env = EmbeddedBuiltinEnv(launcher, output, learner_player=0)  # absolute output; engine loads once
episode = collector.run_episode(env, episode=1, seed=7)      # unchanged BuiltinCollector
```

Later environments in the same process reuse the loaded engine. Parallelism is
one engine per worker process.

## Validation

Asset-free tests: `tests.test_embedded_engine` (codec equivalence, refusals,
lifecycle without assets, native channel contract), `tests.test_embedded_env`
(collector, both seats, natural/cutoff/failure, consecutive games, evidence
rules, replay identity) and `tests.test_validate_embedded_engine`.

Real games use `validation/probes/validate_embedded_engine.py`, one scenario per
process, with matched cold and embedded rows:

```sh
P="python -m validation.probes.validate_embedded_engine run --binary vcmi/build/bin/vcmiclient \
  --replay-binary vcmi/build/bin/vcmireplay --data <Data parent>"
$P --backend cold     --controller policy --checkpoint <frozen.pt> --map <map> --episodes 8 --output <new>
$P --backend embedded --controller policy --checkpoint <frozen.pt> --map <map> --episodes 8 --output <new>
$P --backend embedded --controller visit --visit-kind 101 --seats 0 --map experiments/explorer/maps/dialog-chest-red-v1.vmap ...
$P --backend embedded --controller end_turn --episodes 20 ...           # repeated resets, memory and threads
$P --backend embedded --controller stall --bridge-timeout 5 --stall-seconds 8 ...  # poisoning
python -m validation.probes.validate_embedded_engine compare --left <cold> --right <embedded> --output <file>
python -m validation.probes.validate_embedded_engine codec --binary vcmi/build/bin/vcmiclient --episode <cold episode>
```

Every row is replayed independently unless `--replay` says otherwise. Policy
rows also reconstruct the frozen policy's choices from the persisted history.

## Limits

- One engine per process. VCMI's `LIBRARY`, `GAME`, settings, logging and module
  state are process-global, so no independent multi-world or multi-thread claim
  is made.
- A native abort inside an AI task terminates the worker process. Run workers
  under a parent watchdog; the closing join is not time-bounded.
- With `until_boundary`, the step watchdog measures time to the next delivery
  batch, so it can fire up to about twice its nominal value.
- The working directory and profile environment belong to the engine.
- Retained loaded resources are reused, not certified transitively immutable;
  repeated-reset memory and thread checks are bounded evidence, not a proof of
  the absence of leaks.
