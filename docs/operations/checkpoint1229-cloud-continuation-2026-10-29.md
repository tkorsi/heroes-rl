# Checkpoint 1229: cloud lifetime and continuation preparation

Reviewed on 29 October 2026. This document prepares a possible continuation; it
does not establish cloud readiness or authorize a training launch. The user
subsequently authorized one separate cloud probe chat and one inspection
follow-up. Both completed without launching a probe or training process; the
setup limitation is recorded below.

## What the official documentation establishes

Codex Cloud tasks run remotely and can continue while the user's computer sleeps.
Each task starts with its own workspace from a published environment. This
removes dependence on an awake Mac during an active cloud task. See the
[Codex Cloud overview](https://learn.chatgpt.com/docs/cloud) and
[cloud environment documentation](https://learn.chatgpt.com/docs/environments/cloud-environments).

Existing tasks retain saved files and installed tools. Saved VM state is
recoverable by default for up to seven days after starting a turn or resuming a
task. Saved state does not establish continuous CPU execution or a surviving
operating-system process. Default Pro/Business/Enterprise task resources are
four vCPUs, 16 GiB RAM and 32 GiB disk; Plus defaults are two vCPUs, 8 GiB RAM and
8 GiB disk. Exact access and capacity must be checked for the selected account.
See [saved state](https://learn.chatgpt.com/docs/environments/cloud-environments#reuse-and-update-saved-state)
and [VM specifications](https://learn.chatgpt.com/docs/environments/cloud-environments#vm-specifications).

Tokens include context, files, tool results and responses. Usage also depends on
model, reasoning, tool use and caching; cloud tasks may consume more allowance
than local messages. Credit-based plans publish token rates. Active turns may
continue after reaching usage limits, subject to fair-use restrictions; this
does not guarantee a particular runtime. See
[usage limits](https://learn.chatgpt.com/docs/pricing#what-are-the-usage-limits-for-my-plan),
[tokens and credits](https://learn.chatgpt.com/docs/pricing#what-are-tokens-and-credits)
and [active-turn limits](https://learn.chatgpt.com/docs/pricing#what-happens-when-you-hit-usage-limits).

The official sources reviewed do **not** establish:

- Detached shell-process survival after an agent's final response, task
  completion, unloading, VM suspension or saved-state restoration.
- A maximum uninterrupted task/command lifetime or an idle-suspension interval.
- Guaranteed execution of a ten-hour blocking command.
- Exact credit treatment of idle shell waiting, or a promise that waiting is free.

The [experimental background-terminal API](https://learn.chatgpt.com/docs/app-server#clean-background-terminals)
documents terminal inspection and termination for app-server. It is not a
hosted Codex Cloud process-lifetime contract. Neither `nohup`, `tmux` nor the
seven-day recovery period proves continuous hosted training.

## Minimize model usage

The recommended design is a standalone, bounded, deterministic training
supervisor. It should make no model/API calls during game collection or PPO.
The script should own deadlines, seed consumption, complete-batch admission,
checkpoints, process cleanup and private publication. Write compact status and
final/failure receipts, and use an agent for preparation, an actionable failure
or final review. Reducing model turns and tool output should reduce tokens;
this is an engineering inference, not a zero-cost-wait guarantee.

For reliable unattended execution, prefer a persistent compute host with a
controlled service/job lifetime and durable private storage. Codex Cloud can
run commands during a task, but the reviewed documentation does not certify it
as an always-on detached training host. Any future cloud preparation must also
validate Linux binaries and runtime acceptance, full-state restoration, replay,
disk capacity and private artifact access. The accepted Mac binaries are not a
Linux runtime. No existing scientific or integrity gate should be weakened.

## Checkpoint 1229 and the local prelaunch check

The failed Mac session remains historical evidence. Checkpoint 1229 contains
134 newly admitted complete PPO updates and 536 training games from that
session. The original target has 2,400 admitted games and **2,600 remaining**;
**869** is the next unused training sequence. The interrupted periodic Blue
evaluation supplies no training data. See the
[sleep investigation](../research/local-sleep-interruption-2026-10-29.md).

The prepared recovery wrapper is **Mac-only**. Its prelaunch command is:

```sh
cd /Users/orsi/Projects/Heroes
./continue-original-1229.sh check --hours 1
```

This is check mode, not a training command. The actual main checkout passed
one-hour and ten-hour checks, and a separate frozen release passed the ten-hour
check. All restored checkpoint 1,229 exactly, calculated 2,600 remaining games
and selected unused sequence 869. See the
[validation record](../../validation/checkpoint1229-recovery-2026-10-29.json).
A later process audit
does not retroactively repair the original deadline/cleanup receipts. Recovery
must bind the immutable failure evidence, exact complete checkpoint state,
admitted lineage and consumed-seed ledger before a separate manual continuation.

## Harmless cloud lifetime probe

[`tools/cloud_lifetime_probe.py`](../../tools/cloud_lifetime_probe.py) uses only
the Python standard library. It makes no network/model calls, launches no child
process, does not detach or restart itself, and touches only a fresh output
directory. It defaults to 120 seconds, accepts at most 1,200 seconds, and exits
when either its wall or monotonic deadline expires. A suspended process cannot
exit while suspended; after resumption it records the overdue observation and
exits before another sleep.

For a future explicitly authorized cloud test, use a new task and a unique
directory outside every training artifact root. A foreground invocation is:

```sh
python3 tools/cloud_lifetime_probe.py --output /tmp/heroes-lifetime-UNIQUE --seconds 120 --interval 5
```

Replace `UNIQUE` with a new identifier. Existing directories are rejected. To
test the specific claim that a process survives an agent's final response, the
manual shell task may launch this same bounded probe as a child, recording its
PID/output path. End that agent turn while the probe is still running, then
inspect the same task after the configured duration. This protocol is for a
future authorized test; it has not been performed in Codex Cloud. Do not create
an automatic continuation, restart or recurring polling loop.

Inspect `session.json`, every `heartbeats.jsonl` entry, `progress.json` and
`completion.json`. Compare timestamps with the independently recorded task-end
and revisit times. Check the recorded run ID/PID, both elapsed clocks, gaps and
final status. Missing completion means incomplete evidence. A completion written
only after reopening, or a large heartbeat gap, is compatible with suspension;
persisted files alone do not demonstrate uninterrupted execution. Gaps may also
reflect scheduling or clock changes. Even uninterrupted observations over this
short probe are only observations, not a platform SLA or a ten-hour guarantee.

The probe is unrelated to checkpoint admission. Its output cannot satisfy a
training recovery, runtime acceptance or publication gate.

## Actual cloud attempt

The authorized cloud chat, titled `Probe Launch Blocked`, was created as
conversation `6ae29a37-5b44-83eb-85ea-1a6d208c1f07`, with executor task
`01a1852a-b5d6-7768-a17a-ce34303db06b`. It stopped before download or execution.
The managed-cloud runtime skill at
`/root/.codex/plugins/cache/openai-curated-remote/cloud-environment/0.1.2/skills/runtime/SKILL.md`
requires using `cloud_environment.environment_status` at startup and inspecting
the selected executor's `/etc/codex/network-policy.json` before network work.
The status tool was not exposed, and the policy file was absent.

The single authorized inspection follow-up confirmed a Linux x86_64 executor
with Python 3.12.14, no exposed managed-cloud readiness/network-policy tool, no
probe process in `/proc`, and no probe outputs in the workspace or inspected
temporary directories. Classification: **setup blocked, no probe launched,
lifetime unmeasured**. This is not an observed process-survival failure. No
download retry, replacement probe, configuration change or recurring model
supervision was attempted. The cloud continuation remains unaccepted until a
properly configured executor and an independently validated Linux runtime are
available; Mac recovery readiness does not satisfy either condition.

Local validation on 29 October used a fresh temporary directory and a real
0.4-second run: five observations, no recorded gaps, deadline completion and
matching progress/completion records. Reusing that directory was refused without
changing its files. Four invalid duration inputs were rejected before creating
output. Three synthetic clock cases (wall expiry, monotonic expiry, and a
backward wall jump with monotonic expiry) each stopped after one wait and
recorded the discontinuity. These are probe checks, not cloud-lifetime or
training validation. No official artifact was changed.

## Subsequent explicit launch request

The user later authorized a new cloud chat to start a ten-hour script. That
authorization supersedes the earlier preparation-only scope, but does not make
the Mac package Linux-ready. The new chat `Setup blocked report` stopped before
download or launch because its executor again lacked the managed status tool
and required network-policy snapshot. No training window started. The
[attempt record](../../validation/cloud-launch-attempt-2026-10-29.json) records
current resource observations and the unchanged Mac checkpoint audit.

The current runner has only host-local flock exclusion. Establish an enforceable
exclusive handoff before any remote official game, and separately validate full
lineage relocation and actual Linux runtime binaries. The app's computer-use
policy denied opening its own settings, so environment configuration could not
be repaired from the local task. Supported environment setup is documented in
[Cloud environments](https://learn.chatgpt.com/docs/environments/cloud-environments#create-and-publish-an-environment).
