# Experimental native EngineSession

This Optimizer work starts from PR #25 merged at `875841a` on
`codex/builtin-ppo-pilot`. The new branch is `codex/portable-engine-session`.
It includes the requested Teacher native correctness source at `65a4f48`, but
changes neither its active Python runtime nor checkpoint/resume guards. Python
inference/performance and PPO remain in their respective ownership lanes. No
training was run. The opt-in C++ live slice now has bounded real-game validation;
it is not a general certification of all VCMI retained state.

## Dependency map

| Dependency | Actual responsibility | Session treatment |
| --- | --- | --- |
| `lib/gameState`, `lib/mapObjects`, `lib/bonuses`, pathfinding and spell rules | Actual Heroes state and rules | Preserve the pinned VCMI implementation; no replacement simulator |
| `server/CGameHandler`, processors, queries and battles | Authoritative action validation and transitions, RNG, turn and automatic combat boundaries | Use the existing live server path unchanged |
| `CClient`, packet visitors, per-player callbacks | Client state updates, callback delivery, ACK ordering and terminal barrier | Preserve existing live path and locks |
| ExplorerRL contextual/visible-state code | Player-filtered history, contextual action availability and native callback requests | Same implementation; only message transport is injected |
| Nullkiller and BattleAI | Fixed native adventure opponent and automatic combat | Same modules; Nullkiller Easy, no cheats and normal fog |
| `GameLibrary`, resource/filesystem loaders, mods, settings, dynamic AI modules | Content loading and process-wide services; some mutable caches/statics | Load once in a session; audit and real reset tests still required before treating retained state as safe |
| Upstream `ServerThreadRunner` and `NetworkHandler::createInternalConnection` | Existing in-process live server/client transport | Select thread/internal connection mode, not a fork or a replacement rule executor |
| Added ExplorerRL POSIX socket/poll/recv/send glue | External policy transport, not game rules | Extract into `SocketPolicyTransport.cpp`; default cold-client fallback preserved |
| Added macOS `VCMI_USER_DATA_DIR` in `VCMIDirsOSX` | Profile/resource path selection | Host configuration and provenance; not platform-dependent game rules. Linux uses existing XDG paths |
| Optional replay `ForkEpisode`, `prctl`, `/proc`, process groups | Linux replay-job isolation, supervision and inherited file-position repair | Remains separate; not used by EngineSession and not a live-game kernel |
| Native capture/replay and runtime inventory | Evidence, provenance and independent acceptance | Required separately; a channel receipt or a completed command script does not verify gameplay |
| Teacher Python resume guards | Runtime/version/native-byte/path admission for checkpoint continuation | Outside this branch; no claim that tensors or game rules are inherently Mac-specific |

The VCMI library is already portable native code with platform-specific build
and filesystem services. Its multithreaded AI and inherited RNG paths are not
promised to produce bit-identical future games across compilers or operating
systems. Keep the actual seed, source/build/data identity, controller mapping and
recorded events; test each capture with its matching native verifier. A seed
alone does not capture all Nullkiller/BattleAI thread-local randomness.

## Implemented live slice

`engine/NativeSession/EngineSession.h` exposes `load`, `reset`, `observe`, `step`,
`waitForShutdown` and `closeEpisode`, with typed JSON messages and generation
identities. It has no socket descriptor, fork mode, PID or `/proc` contract.
The API is C++20 and uses the existing VCMI `JsonNode`; it is not a new policy,
observation schema, C ABI or Python binding.

The implementation owns one process exclusively because VCMI's `LIBRARY`,
`GAME`, settings, diagnostics and module state are still global. It refuses a
second concurrent owner or attachment to an active client. Resources may be
loaded only once in a process. Reuse happens through the original owner's
`reset`, not by reconstructing `GameLibrary` or silently retrying a failed load.
A native failure poisons reuse; fall back to a fresh ordinary process.

`load` initializes the actual filesystem and game library once. `reset` creates
a fresh `GameInstance`, client, authoritative server/handler, game state, AI
objects, callbacks, request IDs, sockets/internal connections and worker threads
through the existing headless startup path. It selects the upstream
`ServerThreadRunner` with internal connections rather than `ServerProcessRunner`.
It sets the existing v7 single-learner/Nullkiller controller profile, fixed Easy,
BattleAI, no cheats and normal fog. Every action still passes through the real
ExplorerRL/CCallback/server validation and client packet callbacks. No action,
rule, reward, RNG algorithm or terminal-origin verifier is replaced.

An episode requires a unique name and new absolute capture directory. The host
sets/restores only its own episode environment selectors after the previous
threads have joined. The existing native and contextual ledgers are produced in
that directory, including private authoritative data. Do not upload or commit
them. The host does not change the persistent profile/data search path: supply
that before loading and keep it stable. Source builds must run from their normal
`vcmi/build/bin` working directory so development resources resolve correctly.

The typed `InProcessPolicy` lives in `libvcmi`, so the host and dynamically
loaded ExplorerRL share exactly one registration. A player/episode can claim it
once. It transports only the messages ExplorerRL already sanitizes for that
player, without serializing them to a socket. Replay-only engine metadata stays
in existing ledgers. Both directions copy owned JSON values; no engine pointer
or borrowed observation crosses the API. Native inference remains the caller's
responsibility and is outside measured `step` time.

`observe` returns the ordered intervening messages when the adapter reaches its
next response wait, or a terminal. Until submission it returns the same held
boundary. `step` requires that exact generation/decision token and the unchanged
adapter response envelope. It rejects a duplicate, stale, closed or terminal
token. A terminal supersedes a held decision with a new token and does not
redeliver already observed messages. One step means a policy response through
the next adapter response wait, which may be a dialog rather than a whole turn.
The existing adapter still validates action availability, response identity and
server outcomes. Submission is not evidence of an accepted game action.

The queue is bounded at 64 messages. Overflow fails explicitly instead of
blocking a native callback while it owns event/game-state locks or dropping a
visible event. `close` wakes a pending response wait. Host-requested truncation
is unscored; it suppresses the expected channel-cancellation exception, not a
native gameplay error. Terminal observation alone is not completion:
`waitForShutdown` requires the existing successful native barrier/shutdown
receipt; independent replay is still required for acceptance.

`closeEpisode` closes the channel, joins the existing client/network and server
lifecycle, destroys `GAME`, removes the player binding and explicitly detaches
and releases the contextual recorder. Only after that may the new host reset
episode diagnostic failure/packet/receipt counters. The recorder release has a
native ownership test; it is not evidence of every VCMI cache being reset.
The normal client/replay paths do not call the new quiescent-reset/release hooks.
The default socket backend keeps loopback JSONL semantics, message identity,
strict parser and response limit; send/read deadlines now use absolute bounds
across partial operations. This is transport behavior, not a rule change.

## Build and deterministic admission checks

Use the existing dependency setup, then:

```sh
python3 tools/prepare_vcmi.py --check
cmake --build vcmi/build --target vcmisession native_session_contract \
  vcmiclient vcmireplay --parallel 2
python3 -m unittest tests.test_native_session tests.test_native_contextual_evidence \
  tests.test_optimizer_lifecycle tests.test_optimizer_benchmark \
  tests.test_optimizer_inventory -v
```

Tests compile/use actual production channel, socket and evidence code. The
100-generation fixture checks owned observations, per-player separation,
stale/duplicate tokens, terminal cancellation, bounded capacity, response
limits/deadlines, binding removal and actual diagnostic-counter reset. Another
100 unloaded host lifetimes check exclusive ownership and fail-closed API
admission. The legacy socket fixture uses a temporary loopback peer and no
simulated game protocol; it requires permission to create localhost sockets.
These deterministic tests contain no game Data, fabricated verified replay or
mock Heroes rules. They do not establish live fog/RNG/reset equivalence.

## Native command harness and reproduction

`vcmisession --commands /private/commands.jsonl --output /private/new-result`
is a native integration harness for explicit caller-selected actions. Commands
are newline JSON objects; receipts and a final summary go to files so startup
logs cannot corrupt a wire protocol. It never chooses a policy action or runs
training. A minimal real-input command sequence starts like this:

```json
{"kind":"load"}
{"kind":"reset","episode":"A-cold","map":"Maps/your-installed-map.vmap","seed":12345,"learner":0,"capture_directory":"/private/new-A-cold-capture"}
{"kind":"observe"}
```

Subsequent `step` commands carry `response` with the exact existing v7 contextual
action envelope (episode, player and decision identity included). The harness
uses the last held token, while the C++ API requires it explicitly. Read the
real observation before selecting a response; do not invent an action trace.
`{"kind":"await_shutdown"}` waits for a real natural barrier, and
`{"kind":"close"}` releases the episode. An explicit cutoff closes without
inventing a natural outcome. A completed script is labelled
`commands_completed_unverified`, with `replay_verified: false` even when the
native shutdown succeeded. Unknown/failed commands stop without replacement.
This is an incremental native host, not a replacement for Teacher collection,
checkpoint portability or Claude's inference/performance runner.

On the source build, with a legally populated isolated profile already selected:

```sh
(cd vcmi/build/bin && ./vcmisession \
  --commands /private/commands.jsonl --output /private/new-result)
```

A file containing only `{"kind":"load"}` is a real resource preflight probe.
Without usable Data it must fail, retaining `summary.json`, not claim readiness.

## Real cold/reused A/B/A gates

Keep a reviewed inventory of every selected Data/config/mod/plugin path and an
isolated immutable runtime using the PR #25 inventory command. The inventory
binds declared bytes at boundaries; it does not establish complete dynamic-loader
coverage or prove that all retained content is immutable. Record any enabled
mods; unsupported or unmeasured configurations cannot inherit acceptance.

Prepare three bounded episode schedules A/B/A using existing legal map inputs,
both learner seats, distinct seeds/maps, real natural and prefix endings, and
changed contextual controllers only if supported by a separately reviewed host.
This host currently supports the fixed v7 learner/Nullkiller profile only.
Use the same explicitly selected policy implementation, limits and workload in
both modes; inference is outside native advance timing. A seed is not a promise
that native opponent scheduling produces identical games. Account for any action
trace divergence explicitly rather than forcing an illegal recorded response.

For cold mode, run one new `vcmisession` process per episode command file, each
with `load/reset/observe/step/.../close` and a unique capture/output directory.
For reuse, run one process with a single `load`, then the same A/B/A schedules
with fresh `reset/.../close` commands and unique episode/capture identities.
Repeat in alternating cold/reused order; do not replace failed rows. The native
API supports an external controller reading each real boundary; the fixed-file
harness is useful only when its preselected response trace actually remains
valid. It is not an inference campaign or an admission shortcut.

Record load time, reset dispatch plus waits through the first held boundary,
submit-to-next-boundary latency, successful/rejected native actions, terminal or
cutoff reason, cleanup and process peak resident bytes. Timings are cumulative
per session; subtract successive receipt counters for each episode/step. Reset
and step timers exclude caller idle/inference. Peak RSS is a **process high-water
mark**, not current RSS or an episode delta (Linux KiB and macOS bytes are
normalized by private platform glue; unsupported platforms report unavailable).
Inspect retained memory/caches and file/thread cleanup across many episodes;
a high-water metric alone cannot prove absence of a leak. Filesystem, native
join and upstream lobby startup are not preemptible hard real-time guarantees;
the existing startup wait can take 60 seconds. Use an outer bounded process
supervisor and cold fallback if native cleanup does not return.

Independently verify each actual capture with the matching `vcmireplay`, including
its contextual ledger and player-visible messages; use `--prefix` only for declared
cutoffs. Then apply the existing reviewed cold/worker replay comparison to each
capture's A/B/A replay jobs. That replay comparison verifies replay-mode reports;
it does **not** establish live-session equality by itself. Inspect player-visible
history, fog, contextual availability, request/packet hashes and complementary
natural terminal/barrier accounting across both live modes. Also compare reused
A against cold A for stale IDs, queries, pending ACKs, fog, caches and memory;
record nondeterministic differences rather than asserting seed-only equivalence.
Acceptance remains scoped to the actual measured workloads below. The ordinary cold headless client remains the portable fallback;
the Linux replay fork worker is optional and independent.

## Measured Linux milestone (2026-10-25)

Private original Data was reconstructed with whole-file hashes outside the public
source tree. No assets or private captures are committed. Compact results are in
[`validation/native-engine-session-2026-10-25.json`](../validation/native-engine-session-2026-10-25.json).

The final cohort used three alternating-order cold/reused A/B/A repeats: nine
fresh-process games versus nine games in three persistent sessions. A and B use
two authored maps, both learner seats and matched requested seeds/limits. The
explicit controller only ends turns and selects the first displayed modal choice;
it performs no learned inference. All 18 games reached natural native shutdown
and independently passed full native/contextual replay, with zero unchecked
items. Each mode checked 4,662 client packets. Original failed trials remain
failed and unscored; this is a separately identified new cohort.

| Aggregate, nine games per mode | Fresh process | Reused session |
| --- | ---: | ---: |
| Process wall seconds | 3.888 | 2.356 |
| Resource load seconds | 1.923 | 0.634 |
| Reset through first boundary seconds | 0.452 | 0.415 |
| Native advance seconds (inference excluded) | 1.106 | 1.128 |
| Cleanup seconds | 0.043 | 0.039 |
| Maximum peak RSS MiB | 78.46 | 81.36 |

The aggregate wall ratio is approximately 1.65, predominantly from amortizing
resource loading. Filesystem caches were warm: cold means a fresh process, not
cold storage. This tiny end-turn fixture does not establish training throughput
or faster game rules. Native advance was approximately unchanged. Timing uses
blocking process reap and an independent watchdog, avoiding timed-wait polling
quantization. Declared Data/Maps/config/mod/plugin trees and runtime identities
were checked before/after; the same base profile settings were restored before
each process. Declared inventory coverage is not complete dynamic-loader proof.

A separate persistent process completed 100 consecutive real one-step prefixes;
all independently replayed successfully. Post-cleanup descriptors stayed at 8 and
threads at 5. Current RSS rose from 76.71 to 84.80 MiB; its last 20 readings ranged
84.71–84.80 MiB. This bounded audit does not prove absence of leaks or all mutable
AI caches. Every first private decision matched the cold reference. Raw initial
full-state digests differ across fresh starts: existing `StartInfo` serializes a
wall-clock `startTime`. The verifier retains all original full-state hashes and
checks each capture against its complete recorded inputs; no normalization was
applied. This does not promise identical future autonomous games from seed alone.
A diagnostic attempt to decode StartInfo without initialized game resources
failed and was discarded; it is not supporting evidence for input equivalence.

The direct API also passed a real open-hero/close/move sequence: position
`[4,7,0]` to `[4,8,0]`, movement 1560 to 1460, three actual contextual submissions,
23 independently checked client packets and zero unchecked items. The focused
native regression suite passed 46 tests with no skips.

For reproducible controller-driven runs, `native_session_benchmark --plan
/private/plan.json --output /private/new-result` consumes an object with an
`episodes` array. Each entry has `episode`, `map`, `seed`, `learner`, `max_steps`
and `seconds`. Run each entry in a separate process for cold mode; use one array
A/B/A for reuse. Use new result directories and record actual profile/runtime
identities. The benchmark emits `completed_pending_independent_replay` and
`replay_verified: false`; verify every episode separately:

```sh
./vcmireplay --log /private/episode/engine/replay.jsonl \
  --contextual /private/episode/engine/contextual-replay.jsonl \
  --report /private/new-verification.json
```

Use `--prefix` only for intentional cutoffs. Contextual verification includes
the private observation/message stream; do not combine the separate raw
`--observations` mode with `--contextual`. Raw v7 observation files alone are not
the contextual acceptance gate. No Teacher failed game is retrospectively scored.

The ordinary shutdown fix is separately applicable in PR #27 (`ca488fd4`), with
an explicit additional terminal-attempt accounting contract (`16085b1`). See
[shutdown evidence](native-headless-shutdown.md). The live session remains opt-in;
the ordinary cold client is the fallback. Broader maps/mods, long games, retained
AI randomness/caches, macOS execution and Teacher integration require their own
gates. Portable execution does not promise cross-compiler bit identity.
