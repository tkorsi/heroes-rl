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

`vcmireplay --experimental-worker 120000` loads VCMI rules/resources once, then
accepts one JSON argument array per stdin line. It replies with one JSON ready
receipt and one completion receipt per job on stdout. Diagnostics use stderr.
An example job (paths are placeholders):

```json
["--log", "/private/capture/engine/replay.jsonl", "--contextual", "/private/capture/contextual.jsonl", "--report", "/private/new-output/report.json"]
```

The worker is a Linux-only, sequential fork template. It checks
`/proc/self/task` for exactly one thread before and after library loading and
before every fork. It never constructs a game handler in the parent. Each child
runs the existing replay path with a fresh `Replay`, `CGameHandler`, callbacks,
observation accounting and recorded seed/StartInfo. All episode mutations,
including mutations to library caches and process RNG state, are copy-on-write.
Each child exits without inherited global destructors or settings saves.
Child exit, crash and deadline expiry all lead to reaping; remaining members of
the child's process group are killed. A Linux parent-death signal kills the
episode leader if the template disappears.

Fork also shares open-file descriptions. After every job the parent restores
the positions of inherited read-only regular files, including archives. It does
not rewind diagnostic output. File contents and other external side effects are
not copy-on-write. Use a dedicated isolated VCMI profile, immutable Data/config,
unique report paths and no concurrent writer to that profile or asset tree.
Changing assets/config requires restarting the template. The prototype does not
establish VCMI thread safety, reentrancy or live gameplay-worker safety.

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

The lifecycle test compiles the actual process-reset header and checks 100 fresh
memory/RNG/read-position resets, nonzero exit, exception, crash, deadline,
post-failure reuse, multithread rejection and no unreaped direct children.
These are process-primitive checks, not simulated Heroes gameplay evidence.

With legally supplied Data and **existing** native captures, create a JSON list
of argument arrays. Include A/B/A (different maps/seeds and both seats), at least
one verified natural ending and one verified `--prefix` cutoff. Include a changed
controller mode when available. Each row supplies `--log` and optionally
`--contextual`, `--observations`, `--prefix`; the tool supplies unique reports.
Then run under the same isolated profile used by the normal replay binary:

```sh
python3 tools/benchmark_native_lifecycle.py \
  --binary vcmi/build/bin/vcmireplay --jobs /private/jobs.json \
  --output runs/native-lifecycle-ab --repeats 3
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

GitHub API access from this runtime returned `Forbidden` for issue/PR operations;
read-only GitHub pages supplied coordination context. If posting remains blocked,
the branch and this document are the reviewable handoff, and issue #4 / a draft
PR still need publication once write access is available. Do not message or
redirect Claude, or launch anything on the user's Mac.
