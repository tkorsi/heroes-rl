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

Initial prototype validation, before the v2 service below, built all six native targets (`vcmiclient`, `vcmiserver`,
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

The asset-free v2 batch-service milestone rebuilt `vcmireplay`, passed exact
recipe agreement and all **14 focused tests** (8.403 seconds, no skips), and
preserved the ordinary one-shot usage behavior. These are the current service
checks; the earlier full suite was not rerun. Detailed coverage and cleanup
limits are in [batch-service evidence](../validation/optimizer-batch-service-2026-10-25.json).
The real resource probe still exits before readiness for missing `DATA/VCDESC`.

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
python3 -m unittest tests.test_optimizer_lifecycle tests.test_optimizer_benchmark -v
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
directly linked library bytes and rejects drift. The optional explicit runtime
inventory below additionally binds declared Data, configuration, mods and extra
library files. It cannot certify that these are all inputs used by the loader.
Failed or incomplete runs cannot acquire an equivalence label.
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


## Explicit runtime inventory and admission

The replacement cloud Optimizer continues the same branch/PR from `a6414d8`.
This opt-in extension is limited to this native replay benchmark. Existing
one-shot and service protocol behavior is unchanged. No private game inputs
were supplied by this milestone; temporary test files are admission fixtures,
not replay captures or substitute Data.

Create an untracked `/private/runtime-spec.json` using **absolute canonical real
paths** from the actual runtime installation. All four lists are mandatory;
`data` and `configuration` must each contain at least one existing nonempty
file/tree. Explicit empty `mods` and `additional_libraries` lists mean that no
such additional inputs are declared, not that their absence was measured.
For example (replace every placeholder):

```json
{
  "schema": "heroes-native-runtime-inventory-v1",
  "data": ["/private/heroes-runtime/Data", "/private/vcmi-runtime/config"],
  "configuration": ["/private/isolated-profile/config/settings.json"],
  "mods": ["/private/vcmi-runtime/Mods"],
  "additional_libraries": ["/private/vcmi-runtime/AI/libBattleAI.so"]
}
```

List every resource/search root and configuration input actually selected by
this installation, including engine resource/config trees outside the proprietary
Data directory. Review enabled mods and plugin paths manually. Use the same
isolated profile and exported path variables for inventory and execution. Keep
mutable logs/cache/report directories outside inventoried roots. The worker must
not save changes into these declared immutable inputs. A deliberately empty mod
tree can be omitted via `"mods": []`; a declared empty root is rejected to catch
accidental placeholder installations. File symlinks bind both link text and real
target bytes. Directory symlinks are refused with an actionable error: declare
canonical targets as separate roots and use a real directory layout. Special
files are refused; no FIFO/device is opened for hashing.

The commands below use the source build's `vcmi/build/bin` working directory,
which selects its development resource paths. For an installed runtime, use its
actual resource-selecting working directory for both commands. A root-directory
launch without installed resources can fail on `CONFIG/FILESYSTEM` before game
Data is inspected. First generate and review an inventory without launching a
native process:

```sh
(cd vcmi/build/bin && python3 ../../../tools/benchmark_native_lifecycle.py \
  --inventory-only --runtime-spec /private/runtime-spec.json \
  --output ../../../runs/native-runtime-inventory)
```

`inventory.json` retains all missing/unreadable declared paths and role errors;
exit 2 means admission failed. Fix the local inputs and choose a **new** output
path before retrying. Hashes, names, file sizes, executable bits, empty directory
membership, symlink targets, working directory and selected path variables
(`HOME`, XDG config/data/cache roots and `XDG_DATA_DIRS`, `LD_LIBRARY_PATH`,
`LD_PRELOAD`, `VCMI_USER_DATA_DIR`) are retained. No general environment or credential dump occurs.
Treat local inventories as private: they contain installation paths. They are
not intended for committing/uploading with copyrighted captures/assets.

For A/B/A, write `/private/jobs-aba.json` as a JSON list of the same argument
array A, a different array B, and A again, using existing native captures.
A and B must cover different seeds/maps and both seats; include verified natural
termination and a `--prefix` job and changed controller mode when available.
Do not invent captures to complete this matrix. Then run:

```sh
(cd vcmi/build/bin && python3 ../../../tools/benchmark_native_lifecycle.py \
  --binary ./vcmireplay --jobs /private/jobs-aba.json \
  --runtime-spec /private/runtime-spec.json \
  --runtime-manifest ../../../runs/native-runtime-inventory/inventory.json \
  --output ../../../runs/native-lifecycle-aba-inventoried --repeats 3 \
  --startup-seconds 120 --job-seconds 120 --overall-seconds 3600 --cleanup-seconds 2)
```

The reviewed inventory is required alongside the spec; an ordinary comparison
cannot silently regenerate its expected identity. Preflight rejects missing
captures, empty observation evidence, unresolved linked libraries, unavailable
runtime inputs or drift before launching either replay mode. It rechecks binary,
capture/job bytes, observation membership, directly linked libraries, declared
runtime trees, selectors and the spec/manifest bytes before every mode launch
and every individual cold/worker dispatch. It rechecks again before accepting
each comparison and final equivalence. A mismatch stops without retry and
retains failed `summary.json` plus any existing native/receipt logs. Admission
checks retain stage, timing and error; runtime drift also retains expected and
actual inventory and changed path names. Missing capture failures retain their
exact error even before runtime admission. Completed service receipts survive a
later admission failure and undispatched jobs stay `not_started`.

These checks establish **declared bytes at sampling boundaries**, not a complete
runtime provenance proof. Preflight cannot prove that VCMI will find logical
`DATA/VCDESC` inside archives, validate every asset's format, establish successful
resource loading, or infer undeclared search roots. Additional library paths are
operator-declared, not measured `dlopen` coverage; `ldd` covers resolved startup
libraries only. There is no open-file trace, loader namespace audit, hash binding
of every host environment variable, or protection against a concurrent writer
that changes and restores bytes between checks. Maintain exclusive access to an
immutable runtime for the whole campaign. Native readiness and existing replay
verification remain separate mandatory gates. Inventory hashing is synchronous
and may be expensive for real Data: recorded wall times include this admission
I/O, while native phase timers remain unchanged. Overall deadline checks reject
late admission before dispatch/acceptance, but blocking filesystem operations
are not preemptible wall-clock guarantees. Use native phase measurements and
account for admission overhead before making any performance claim.

Remaining real acceptance prerequisites: legally supplied usable game Data,
complete selected VCMI resources/config/mods/plugin paths, an isolated stable
profile, and actual A/B/A native capture/contextual/observation files covering
the matrix above. The prior native probe failed before ready on missing
`DATA/VCDESC`. This milestone provides no real reset-equivalence or speedup
result and does not integrate with the Teacher or a second gameplay verifier.

Validation for this extension: **24 focused tests pass, zero skips**; `vcmireplay`
builds, the repeat build has no work, and exact recipe agreement passes.
The source-build native probe emits no ready receipt and exits 3 for missing
`DATA/VCDESC`. Compact measured evidence, temporary-fixture scope and remaining
prerequisites: [runtime inventory validation](../validation/optimizer-runtime-inventory-2026-10-25.json).
