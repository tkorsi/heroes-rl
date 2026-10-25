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
and no training was run. Measurements and real-game acceptance on macOS and
Linux are recorded in
[embedded engine validation](../validation/embedded-engine-2026-10-25.json),
including which code revision each platform ran.

## What runs where

| Concern | Cold backend | Embedded backend |
| --- | --- | --- |
| Game rules, state, RNG, routing, queries, battles | `vcmiclient` process with its in-thread server | Same VCMI code in the Python process (`EngineSession`, in-thread server, internal connection) |
| Player-visible stream | ExplorerRL, JsonWriter line over TCP, Python reader thread, `json.loads` | Same ExplorerRL records, handed over in memory and converted directly to Python values |
| Learner response | `json.dumps` line, VCMI strict JSON parser | Same line and the same parser (`submit_text`); 64 KiB admission kept; a refusal is recorded differently (see Learner contract) |
| Opponent | Nullkiller Easy created by the client's AI loader | `NullkillerOpponent`: the same module, created per game through an installed provider, released and reset after the game |
| Automatic battles | BattleAI for both sides | Unchanged |
| Episode boundaries | Process exit status or SIGTERM/SIGKILL | `finish()` report: exit status 0 after the terminal barrier, or a clean host stop |
| Evidence | Engine capture files, bridge ledger, history, manifest | Same files and schemas, plus the close report; replayed by the unchanged `vcmireplay` |

The player channel is `InProcessPolicy` (`engine/NativeSession/PolicyTransport.*`),
compiled into `libvcmi`. The socket transport
(`engine/ExplorerRL/SocketPolicyTransport.cpp`, built into `libExplorerRL` and
`vcmireplay`) is unchanged and remains the cold path. `InProcessPolicy` existed
before; this branch changes its semantics (incremental `take`/`drain`, terminal
supersession, a 1024-message queue for engine sessions). The cold client's
`libvcmi` carries that code but never instantiates the channel.

## Native API

`EngineSession` (C++20, `engine/NativeSession/EngineSession.h`):

- `load()` loads the filesystem, mods and game library once per process.
- `reset(EpisodeConfig)` creates a new game: settings baseline restored,
  per-game environment, seed, learner seat, capture directory and engine log,
  then a fresh `GameInstance`, server, game state, ExplorerRL and Nullkiller
  instance through the existing headless start path.
- `take(deadline, untilBoundary)` delivers every new player message once, plus
  a boundary token when a decision is newly offered, the terminal latch and the
  first failure. A high-water mark (half of the 1024-message queue) wakes a
  waiting host during a long opponent turn. There is no backpressure: `send`
  never blocks a packet callback that owns game locks, so if the host does not
  return to `take` before the queue fills, the game fails with "Native policy
  message queue overflow" and the engine is poisoned. The socket backend instead
  blocks its sender while the reader thread drains. No validation game came near
  the limit.
- `submit(token, response)` answers exactly the offered decision and returns
  true. When the player's own terminal superseded exactly that offered decision,
  it returns false, once: like a socket line written after the peer's terminal,
  the response is admitted and never applied. Other stale or duplicate tokens
  are refused.
- `finish(natural, deadline)` waits for the natural terminal barrier when asked,
  then always joins every game thread, destroys the game, releases the opponent
  and returns an `EpisodeReport`. A natural finish whose exit status misses the
  deadline poisons the engine; the cold backend kills its process in that case.
- `observe`/`step`, `waitForShutdown` and `closeEpisode` remain for the existing
  C++ benchmark and contract tests.
- `poisoned()` and `active()`. `timings()` reports cumulative load, reset,
  advance, cleanup and natural finish wait, plus opponent-turn and
  automatic-battle totals. Reset and advance count only time the host waits
  inside `take` or `step` for the engine; ingest, validation, ledger writes and
  inference between calls are excluded. Opponent turns are nested in advance (in
  reset when Red Nullkiller moves before a Blue learner's first decision) and
  battles may overlap both, so these are totals, not a split.

The Python module (`PROTOCOL = "heroes-embedded-engine-v1"`) exposes
`Engine(console_log=False)` with `load`, `reset(episode=, map=, capture_directory=,
seed=, learner=, policy_timeout=30, log_path="")`, `take(timeout,
until_boundary=True)`, `submit(generation, decision, dict)`, `submit_text(generation,
decision, str)`, `finish(natural=, timeout=)`, `timings`, `close`, and the
`poisoned`/`active` properties. `submit` encodes the dict with
`json.dumps(separators=(",", ":"))` and then does exactly what `submit_text`
does; both return the boolean above. Tokens must be integers in `0..2**64-1`
(`TypeError`/`OverflowError` otherwise). If a signal handler raises (a budget
alarm, Ctrl-C) while `take` or `finish` returns, the exception propagates and the
result is kept for the next call, so no delivery or report is lost. The
`poisoned`/`active` properties raise `EngineError` while another thread is inside
the engine. Episode identities may be reused after the previous game closed.
Module functions are `resources()` (RSS, peak RSS, threads, descriptors) and
test or measurement fixtures. The GIL is released while the engine works. One
`Engine` may exist per process.

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

`NullkillerOpponent` owns the opponent explicitly. The requested
initialize/advance/reset/shutdown lifecycle maps onto it as follows.
`initialize()` runs once after load. `attach(player)` starts each game: it
installs the provider, a passive packet observer used only for timing, and the
unchanged Easy and BattleAI selections. There is no separate advance call: the
engine's own turn order advances Nullkiller exactly as in the cold client, and
the host never drives or inspects it. `detach()` and `release()` are the
per-game reset, after all game threads joined: the instance is released, then
patch 0029 returns Nullkiller's module-level caches (pathfinding node storage,
cached hero graphs) to fresh-process state. `shutdown()` runs with the session.
An instance that outlives its game, or a module reset that did not happen
(missing export, static AI build or an error), poisons the session, and close
evidence then fails.

Any native failure, failed join, channel failure, experiment error, natural
finish without its exit status, or failed opponent module reset poisons the
engine: the next `reset` is refused natively (`SessionHealth`, before any
mutation) and by the Python environment, and the worker must start a new
process. Experiment failures are logged as `Experiment failure: <reason>` in the
game log, as in the cold client. Patch 0028 lets an embedded session capture an
exception on the in-thread server thread as an experiment failure; the capture
is switched on only while an `EngineSession` is loaded, so the cold `vcmiclient`
keeps its `std::terminate` path.

The engine changes the Python process for the rest of its life and restores
nothing (it is never unloaded): the working directory becomes `build/bin`
(VCMI's development layout resolves data and libraries against `.`), relative
`sys.path` entries are made absolute, every inherited `EXPLORER_RL_*`,
`VCMI_RECORD_*`, `VCMI_EXPERIMENT_*`, `VCMI_USER_DATA_DIR` and `XDG_*` variable
is removed (including unrelated ones such as `XDG_RUNTIME_DIR`), as the cold
launcher strips its child's environment, and the profile variables are set
(`XDG_DATA_HOME`, `XDG_CONFIG_HOME`, `XDG_CACHE_HOME` on Linux,
`VCMI_USER_DATA_DIR` on macOS).

## Learner contract

The learner sees exactly what the cold backend delivers: the same ExplorerRL
player-filtered records, in the same order, decoded to the same Python values.
Tests compare the direct conversion with the cold codec (JsonWriter text with
newlines removed, then `json.loads`) for floats, escaping, control and invalid
UTF-8 bytes. A message the cold reader would reject (undecodable, a duplicate
field after JsonWriter escaping, or a line over 16 MiB) is reported as the same
protocol failure after the same prefix, and the stream is recorded as
incomplete. Responses go through the cold JSON line, its 64 KiB limit and VCMI's
strict parser, so the same responses are admitted. A refused response fails the
game on both backends but is recorded differently: the cold backend records a
delivered line that ExplorerRL then rejects as an experiment failure, while the
embedded binding refuses it before the engine, so the environment records it as
undelivered (`action_delivery_failed`). The learner's typed actions never
produce such a response.

`EmbeddedBuiltinEnv` (`experiments/headless/embedded_env.py`) is a
`BuiltinHeadlessEnv` with the same reset, step, truncate, dialog cutoff, history,
manifest and finalization. The existing `BuiltinCollector` runs on it unchanged.
Hidden information, window-gated content, typed contextual actions including
`move_to(x, y)`, engine-built routes and natural, cutoff and failure
distinctions are those of the shared ExplorerRL and environment code.

## Evidence, identity and replay

An embedded manifest names `identity.backend`: the backend kind, protocol,
module path and SHA-256, and the SHA-256 of every `engine/NativeSession` source.
Runtime, map and module identity are those of the files the process loaded:
they are hashed once at engine load, and every reset and finish requires their
inventory and stat identity unchanged on disk ("restart Python" otherwise).

The `ENGINE_RECIPE` tuple is unchanged, but `engine/patches` is part of it and
this branch adds patches 0028 and 0029, so recipe digests differ from main.
Patch 0028 touches `client/Client.cpp`, `client/ServerRunner.cpp` and the
experiment hooks in `libvcmi`, which are compiled into the cold `vcmiclient`
and `vcmireplay` as well; patch 0029 adds the module reset export to
`libNullkiller`, the cold opponent. Their cold behaviour is unchanged (the
server-failure capture and the module reset are used only by the embedded
session), but cold binary and Nullkiller digests change. The learner checkpoint
source identity is unchanged.

Replay verification (`experiments/headless/builtin_replay.py`) requires the
captured module (by its recorded name, so a verifier on another Python ABI can
check it) and host sources to equal the current ones, then replays the engine
capture with `vcmireplay` and checks the learner history. Finalization
(`experiments/headless/embedded_evidence.py`) first requires every
backend-bearing field (identity, launch command and backend, transport
variable, close report) to name one backend, then proves:

- a natural ending by exit status 0 after the terminal barrier, a clean close
  and an opponent that was created, released and module-reset;
- a host stop by the absence of any exit status, a clean join, the same opponent
  proof and no failure.

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
runtime's file inventory is unchanged; its RPATH points at `build/bin`. With the
option on, the client and server static libraries are built
position-independent, so the bytes (and replay identity digests) of
`vcmiclient` and `vcmireplay` differ from an option-off build of the same
source. Game rules are unchanged; matched rows in the validation record show
byte-identical learner histories between the two backends.

Platform-specific code is limited to host services: resource telemetry
(`/proc` on Linux, Mach on macOS), the profile environment variable and the
extension suffix. No game-rule code differs by platform. The backend was built
and run on macOS (Apple M4 Max, arm64; the module links Cocoa, SDL2 and FFmpeg,
inherited from the client libraries) and on Linux x86-64; the validation record
says which revision each ran. As with the cold backend, AI thread scheduling
means a seed alone does not promise identical autonomous games across runs,
compilers or systems.

## Use

```python
from experiments.headless.embedded_env import EmbeddedLauncher, EmbeddedBuiltinEnv
launcher = EmbeddedLauncher(binary, data, game_map)          # build/bin/vcmiclient path, Data parent, map
env = EmbeddedBuiltinEnv(launcher, output, learner_player=0)  # absolute output; engine loads once
episode = collector.run_episode(env, episode=1, seed=7)      # unchanged BuiltinCollector
```

One environment runs consecutive games through `reset`, and later
environments in the same process reuse the loaded engine. One engine serves one
prepared runtime, Data root and map per process; a launcher with another is
refused. Parallelism is one engine per worker process.

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
$P --backend embedded --controller recorded --recorded <policy run> ...  # engine and adapter without inference
python -m validation.probes.validate_embedded_engine compare --left <cold> --right <embedded> --output <file>
python -m validation.probes.validate_embedded_engine codec --binary vcmi/build/bin/vcmiclient --episode <episode>
python -m validation.probes.validate_embedded_engine collect --runs <run>... --attach <comparison>... --output <file>
```

A row passes only when its game ended naturally or by a declared cutoff, its
replay verified (every row unless `--replay` says otherwise) and, for policy
rows, the frozen policy's choices were reconstructed from the persisted
history. The stall row passes only when its game failed; in-process the probe
then requires the next reset to be refused both by the environment and by the
native engine. Rows reuse one environment per seat (`--env-per-row` creates a
new one per game). `compare` refuses runs whose map, limits, schedule, controller
or policy differ (a `recorded` run may differ from its source only in its
policy fields), reports unmatched rows, and requires equal status, cutoff
reason and outcomes besides identical histories. `collect` writes a path-free
record that names runs by content digest. Where `vcmi/build/bin` cannot load the
frozen checkpoint (its exact Torch release is bound), `--random-policy-seed`
uses a labelled freshly initialized v7 model instead.

## Limits

- One engine per process. VCMI's `LIBRARY`, `GAME`, settings, logging and module
  state are process-global, so no independent multi-world or multi-thread claim
  is made.
- A native abort inside an AI task terminates the worker process. Run workers
  under a parent watchdog; the closing join is not time-bounded.
- With `until_boundary`, the step watchdog measures time to the next delivery
  batch, so it can fire up to about twice its nominal value.
- The working directory, `sys.path` and environment changes above are permanent
  for the process.
- No backpressure on the player channel (see `take`).
- Retained loaded resources are reused, not certified transitively immutable;
  repeated-reset memory and thread checks are bounded evidence, not a proof of
  the absence of leaks.
