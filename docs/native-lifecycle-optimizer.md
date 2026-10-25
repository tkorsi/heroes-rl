# Experimental native lifecycle optimizer

This cloud work starts at `codex/builtin-ppo-pilot` commit `1b22486`.
Issue #4 and draft PR #22 were read on 25 October 2026. Claude retains ownership
of fixed-weight inference, frame encoding/decoding, reconstruction and
`experiments/perf/`. The Teacher owns collection, PPO and checkpoint contracts.
This branch changes neither area. Linux bootstrap files are copied unchanged
from PR #23 at `96d0972`, without replacing the current learner branch.

## Cost breakdown and evidence boundary

There are no private game Data, checkpoint or captured native replay inputs in
this cloud checkout. Consequently there is **no measured native speedup** and
no native reset-independence acceptance yet. The local eight-game 62.161-second
PPO pilot and Claude's offline reconstruction speedups are different workloads.
Neither supplies a cost breakdown for this runtime.

Cloud validation built all six native targets (`vcmiclient`, `vcmiserver`,
`ExplorerRL`, `BattleAI`, `Nullkiller`, `vcmireplay`) and a repeat build had no
work. Exact prepared-source agreement passed. The full suite ran 984 tests:
980 passed and four skipped. A focused nine-test follow-up passed, including
the native notice parser that had skipped before library linking finished;
the three remaining prerequisites are private Data/captures. The final four
lifecycle/admission checks also pass. All four shared libraries load, and native
CLI/error checks pass. The worker's actual load probe exits 3 with
`Resource with name DATA/VCDESC and type TEXT wasn't found.` and emits no ready
receipt. This is an explicit unavailable prerequisite, not a native episode test.
Compact results: [validation evidence](../validation/optimizer-native-lifecycle-2026-10-25.json).

| Cost | Measurement entry point | Current evidence |
| --- | --- | --- |
| Process startup and exit | Cold subprocess wall time minus internal replay time (residual includes report writing) | Requires real capture/assets |
| Rules/resources loading | `timing_seconds.library_load`; worker ready receipt | Requires game Data |
| Episode reset/init | `summary.timing_seconds.episode_init` and `episode_start` | Requires real capture/assets |
| Native transitions | `native_transition_inclusive` around `handleReceivedPack` | Requires real capture/assets |
| Observation verification | `observation_verification` inside packet application | Requires real capture/assets; overlaps native transition |
| State / contextual verification and input decoding | `state_verification`, `contextual_verification`, `input_decode` | Requires real capture/assets |
| Other verification / trace | Internal replay total less measured phases is an unattributed residual | Requires real capture/assets |
| Pathfinding and automatic battle detail | Not separated by this prototype | Unmeasured |
| Live opponent work, transport, policy inference | Existing Claude end-to-end campaign lane | Unmeasured here |
| Complete-game throughput / end-to-end experiment | Existing validated gameplay runner and Claude campaigns | Unmeasured here |

Timers are inclusive scopes, not additive independent phases. Server-log reading
has its own timer. Timing keys are additive report metadata; expected packet,
state, visibility, contextual-action and terminal verification stays intact.

## Small structural prototype

`vcmireplay --experimental-worker 120000` loads VCMI rules/resources once,
then serves the **incompatible v2 experimental protocol** on newline-delimited
UTF-8 JSON. The sole client is updated together. `WorkerProtocol.cpp` owns the
production parser, validation, wire writes and process-reset calls; `ReplayWorker.cpp`
supplies only real VCMI initialization and the existing replay callback. Native
integration tests compile that same helper with transport/process callbacks,
without initializing or imitating game rules.

All messages carry `schema: "heroes-replay-worker-v2"`. The service first emits
`ready` with finite nonnegative `load_seconds`. Requests have exactly four fields:

```json
{"schema":"heroes-replay-worker-v2","kind":"replay","job_id":0,"args":["--log","/private/capture/engine/replay.jsonl","--contextual","/private/capture/contextual.jsonl","--report","/private/new-output/report.json"]}
```

IDs are integers starting at zero and increasing by one; missing, duplicate,
out-of-order IDs, extra fields and old versions are fatal. Each accepted request
emits `started` (`job_id`, episode process-group leader `pid`), then `completed`
(`job_id`, `exit_code`, `signal`, `timed_out`, `seconds`, `cleanup_complete: true`).
A completion means process cleanup, not verified Heroes replay. A nonzero result,
crash or timeout is explicit; the service can accept a new independent next job,
but the benchmark stops without retry/replacement. EOF between requests emits
`closed` with the exact completed-job count and exits zero.

Requests are bounded to 65,536 bytes **during** byte reading. Invalid JSON,
duplicate fields, empty records, partial records at EOF, invalid arguments or
existing/colliding output files end the service nonzero, with diagnostics on
stderr and no successful `closed` receipt. No request fragment executes. Broken
output pipes fail nonzero; writes are nonblocking with a one-second bound. Idle
input and initialization are bounded by the supervising client, not an internal
service idle timer. Only one request is outstanding in the supported client.

The worker is a Linux-only, sequential fork template. It checks
`/proc/self/task` for exactly one thread before and after library loading and
before every fork. It never constructs a game handler in the parent. Each child
runs the existing replay path with a fresh `Replay`, `CGameHandler`, callbacks,
observation accounting and recorded seed/StartInfo. All episode mutations,
including mutations to library caches and process RNG state, are copy-on-write.
Each child exits without inherited global destructors or settings saves.
Child exit, crash and deadline expiry all lead to reaping; remaining members of
the child's process group are killed. The service is a Linux subreaper and
reaps ordinary adopted descendants from that group, with at most one additional
second of native cleanup before failing closed. The template reserves SIGTERM
for supervising-parent loss. The episode reserves an unblocked SIGUSR2 handler
that kills its entire process group when the template disappears. These handlers
also cover the tested SIGINT/SIGTERM/SIGKILL disappearance of the Python supervisor
while a callback and ordinary descendants are running.

Fork also shares open-file descriptions. After every job the parent restores
the positions of inherited read-only regular files, including archives. It does
not rewind diagnostic output. File contents and other external side effects are
not copy-on-write. Use a dedicated isolated VCMI profile, immutable Data/config,
unique report paths and no concurrent writer to that profile or asset tree.
Changing assets/config requires restarting the template. The prototype does not
establish VCMI thread safety, reentrancy or live gameplay-worker safety.

The cleanup boundary is ordinary descendants that remain in the episode process
group and preserve the reserved signal disposition/mask. Callbacks that call
`setsid`/`setpgid`, block/replace SIGUSR2, or interfere with supervision are
unsupported. There is no universal containment claim: simultaneous external
killing of the template and episode leader, or template death after leader death
before its group cleanup, can prevent the signal cascade. After template death,
remaining zombies belong to the system reaper; tests check that no covered process
remains running, rather than claiming the vanished service can reap them. An
uninterruptible kernel task can also exceed wall-clock bounds. The supervisor
only targets a reported live group after checking its direct-parent relationship
and `/proc` birth time; it does not kill arbitrary or recycled PIDs.

`benchmark_native_lifecycle.py` now reads readiness and receipts incrementally.
Startup, each dispatch/completion and the overall campaign have separate absolute
deadlines, followed by one shared cleanup allowance. It persists all planned job
IDs and statuses before dispatch, raw wire bytes as they arrive, and each accepted
receipt before proceeding. `service.json`, `wire.jsonl` and `native.log` survive
failures; active work becomes `incomplete`, and undispatched jobs stay
`not_started`. SIGINT/SIGTERM record interruption and clean up; SIGKILL leaves the
already-durable incomplete state and receipts, and native parent-death handling
covers an active ordinary group. Cold runs also keep per-job attribution and use
the remaining overall bound. Protocol closure, worker exit and cleanup must all
succeed before reports are read and independently compared. Only fully verified,
matching native reports may receive `verified_report_equivalence`; transport
completion alone never does. There is no silent retry.

The ordinary one-shot CLI remains the default. There is no learner integration,
automatic opt-in, modified game rule, new observation cache, reduced verifier or
claim of complete-game acceleration. Contextual control settings are configured
inside each child by the existing replay implementation. Real alternating-seat
and alternating-controller checks are still required.

## Reproduction and acceptance

Build the normal `vcmireplay` target after preparation, then check the recipe:

```sh
python3 tools/prepare_vcmi.py --check
python3 -m unittest tests.test_optimizer_lifecycle -v
```

The lifecycle tests compile the process-reset header and the actual production
protocol helper linked to the prepared native JSON library. They cover 100 fresh
memory/RNG/read-position resets, nonzero exit, exception, crash, deadline,
post-failure reuse, thread rejection, malformed/oversized/partial input, ordered
correlation, EOF/broken pipes, ordinary descendant cleanup and reaping, parent
loss, supervisor SIGINT/SIGTERM/SIGKILL, durable partial receipts, and corrupt
response rejection. Fixed corrupt transcripts are negative admission fixtures,
not a second successful protocol implementation. Harness callbacks exercise only
transport/process behavior and produce no fabricated verified game reports.
These tests do not establish native Heroes reset equivalence or throughput.

With legally supplied Data and **existing** native captures, create a JSON list
of argument arrays. Include A/B/A (different maps/seeds and both seats), at least
one verified natural ending and one verified `--prefix` cutoff. Include a changed
controller mode when available. Each row supplies `--log` and optionally
`--contextual`, `--observations`, `--prefix`; the tool supplies unique reports.
Then run under the same isolated profile used by the normal replay binary:

```sh
python3 tools/benchmark_native_lifecycle.py \
  --binary vcmi/build/bin/vcmireplay --jobs /private/jobs.json \
  --output runs/native-lifecycle-ab --repeats 3 --overall-seconds 3600 --cleanup-seconds 2
```

The tool alternates cold/worker ordering, requires every native report to verify,
and compares complete report semantics across both modes and all repeats,
excluding only timing metadata. Exact existing state/packet/observation hashes
remain mandatory. It binds binary, job-manifest, capture, observation-file and
directly linked library bytes and rejects drift. Mods, Data and later `dlopen`
libraries still require an immutable external runtime; the tool does not certify
their identity or a clean host load. Failed or incomplete runs cannot acquire an equivalence label.
Native logs and reports remain ignored. The tool measures replay wall time and
per-job replay latency, not learned-policy step latency or full-game throughput.
Do not accept the optimization for training until real repeated-reset equivalence
and the corresponding live observation/fog/action/terminal contracts pass.

GitHub API publication was restored after the user published the exact
`api.github.com` allowlist addition. Restricted networking and the package preset
remain in place; no new credential was introduced. Authenticated issue reads,
REST draft-PR creation and issue-comment creation succeeded. Draft
[PR #25](https://github.com/tkorsi/heroes-rl/pull/25) targets
`codex/builtin-ppo-pilot`; the scope and validation limitations are posted in
[issue #4](https://github.com/tkorsi/heroes-rl/issues/4#issuecomment-6398668855).
The CLI's GraphQL operation returned 503; REST worked after a duplicate check.
The original validation record preserves its historical access failures; the
[publication record](../validation/optimizer-publication-2026-10-25.json)
records the subsequent successful operations. No native validation was rerun or
new game evidence obtained during the access repair. Do not message or redirect
Claude, or launch anything on the user's Mac.
