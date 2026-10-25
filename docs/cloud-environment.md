# Cloud development and real-game handoff

## Purpose and current state

The user requested the entire engineering loop in the cloud: code changes,
tests and real headless VCMI games, continuing while the Mac is offline.
Gameplay training is still disabled. The Debian development build and source/
native regression tests are now validated as recorded below; real Linux games
remain blocked on private inputs. Setup success is not gameplay evidence.

The migration base is `codex/cloud-environment`. The dedicated continuation
branch is `codex/linux-validation-integration`, based on cloud commit `cf42471`
and incorporating reviewed `codex/local-evaluation-v2` through `c29e96e`.
Do not start from the older `main`, overwrite the cloud branch, or interrupt a
Mac checkout. Issue #4 assigns simulation/performance/offline replay to Claude;
this continuation integrates published correctness fixes and Linux execution,
without importing the separate PPO or performance proposals.

At this base, the long frozen-neural cohort stopped after two replay-verified
cutoffs and one replay failure; five games were not started. The builtin cohort
then delivered a day-2 learner defeat but stopped on request-accounting
validation. Its original result remains failed/unscored; seven games were not
started. A separately recorded repaired-verifier diagnostic now reproduces that
ending exactly, without retroactively admitting the original row. These are
historical results, not expected or guaranteed cloud outcomes.

## Cloud environment

Create a private Codex Cloud environment with the GitHub repository
`tkorsi/heroes-rl`, then use `codex/cloud-environment` in the setup/task checkout.
The official setup flow is **Settings > Codex Cloud > Environments > Create
environment**, or **Work in > Cloud > Select environment > Create environment**.
Install and test the environment before publishing it. An environment name or a
successful clone alone does not establish runtime readiness.

The environment needs a supported Ubuntu/Debian Linux CPU runtime, package
installation access, GitHub source/submodule access, and the Python/PyTorch
package registries. No GPU or graphical desktop is required for the current
headless task. Start with two build jobs and one PyTorch thread. Check available
RAM and free disk before building or launching games; record the actual VM
specification rather than assuming a subscription tier. A long Mac episode
produced hundreds of MB of history, and one complete failed-episode artifact
directory occupies about 2.5 GB. Archive required raw results privately before
discarding the VM; commit only compact evidence.

Official references, checked 25 October 2026:

- [Codex Cloud environment setup and resources](https://learn.chatgpt.com/docs/environments/cloud-environments)
- [Cloud tasks and computer availability](https://learn.chatgpt.com/docs/dots/tasks-and-memory)

The desktop tools in the originating chat could not create a cloud environment,
and computer-use access to the Codex application was denied. Environment
creation/publishing is a required user UI step, not a completed migration.

## Bootstrap and source-only checks

Use `tools/setup_cloud_cpu.sh --help` for explicit stages. The bootstrap installs
CPU dependencies, prepares the pinned engine recipe, builds the required native
targets and can run the test suite. It does not start games or a training run.
Unit tests may exercise synthetic optimizer updates; that is separate from
gameplay training. Setup must report its actual Python, PyTorch and platform
identity and preserve the resolved package inventory.

The original `setup_cloud_cpu.sh` Python stage resolves host-specific packages.
Use the tested `setup_cloud_locked.sh` path below on the current Debian cloud
host. Its separate hashed Linux wheel lock does not replace the macOS arm64
`dependencies/learner.lock`. Preserve checkpoint compatibility checks and keep
native build, source tests and real-game validation as separate evidence gates.

From Git alone, a task can inspect/edit authored sources, apply the pinned VCMI
recipe, build the engine and run source/contract tests. Real games additionally
need the private inputs below. Preserve all declared test failures and skips;
repair actual portability issues without replacing native checks with mocks.

## Private inputs

Provision these inside the private cloud environment or through a private
artifact store approved by the user. Never put them in Git, a public download
URL, a shared environment snapshot or a container registry image.

| Input | Required contents and purpose |
| --- | --- |
| Game data root | The user's original `Data/` directory, about 861 MB. Pass its parent directory to the engine's `--data` argument. Music, videos, installers and Mac app bundles are unnecessary for headless play. |
| Fixed current-interface checkpoint | `fixed-policy.pt`, 629,494 bytes, SHA-256 `4043b58bb3dc4eb6169514ca1ed7c1be0b373a5139efe04b95dfeda1832232c1`. Needed to preserve the frozen v7 model and opponent weights. |
| Private input manifest | Record relative file names, sizes, hashes and provenance; keep machine-specific source paths and bulk inventory outside Git. |

The Mac source is the user's VCMI `Data/` folder and the ignored
`runs/adventure-dialogs-2026-10-25/policy-1/fixed-policy.pt` in the
troop-transfer checkout. A local input manifest can record absolute source paths
for the actual transfer; no account credentials or whole user profile are needed.
Only the selected artifacts should be transferred, after the private destination
is known. This preparation does not upload them or create storage credentials.

Old raw Mac captures are optional for source-level diagnosis. They are not
required for fresh Linux gameplay. Fetch a particular capture only if needed;
do not copy every `runs/` directory. Keep the original failed records unchanged.

## Portability and verification

Mac executables cannot be reused on Linux. Prepared execution plans also bind
absolute paths, source identities and native binary hashes. Evaluators compare
the complete Python build string, PyTorch version and model/interface/source
identity. Native replay checks captured client/rules/AI binary identities.

Consequently, the saved Mac plans are not Linux resume plans. Preserve the
checkpoint bytes and provenance; implement and test an explicit portability
procedure if retaining its weights, with a new separately identified Linux
artifact and exact tensor checks. Do not disable the compatibility guards.
If a fresh random model is used for a runtime smoke check, label it explicitly
and do not claim it is the old trained or frozen policy.

After private inputs and native setup are available:

1. Record the exact cloud source commit, dependencies, runtime and data/map hashes.
2. Run `python3 tools/prepare_vcmi.py --check` and the relevant tests.
3. Prepare a fresh, bounded, zero-training Linux validation plan for both player
   colors. Check real collection, player-local observations, cutoff/terminal
   accounting and native replay under that same Linux runtime.
4. Preserve every attempt, including failures. A natural engine outcome that
   fails verification remains separately labelled and unscored in that cohort.
5. Only after portability succeeds, prepare the authorized comparison as a new
   platform-bound plan. Do not restart the stopped Mac schedule under its old
   identity or reuse its elapsed-time budget implicitly.

Keep automatic battles, normal fog, explicit structured interactions and
terminal-only rewards. No scripted strategy, reward shaping, silent retries,
training campaign or extra paid infrastructure follows from environment setup.
The latest cloud continuation supersedes the earlier per-cohort allowance:
**one shared 3,600-second maximum for all new native diagnostics, policy
reconstruction, replay and cleanup in this task**. It authorizes no additional
self-play cohort or training. Development can continue after this clock is
exhausted or while private inputs are missing.

## Setup prompt

Paste this into the environment setup conversation after selecting the repository:

> Prepare a private CPU Linux environment for tkorsi/heroes-rl using branch
> codex/cloud-environment. Read AGENTS.md, README.md and docs/cloud-environment.md.
> Inspect tools/setup_cloud_cpu.sh, install the required dependencies, prepare
> the pinned VCMI recipe and build the headless engine, ExplorerRL, BattleAI,
> Nullkiller and vcmireplay. Run the relevant tests and report actual resources,
> versions, skips and failures. Create a tested Linux dependency lock. Do not
> download game assets from third parties or place private assets in Git. Request
> the user's Data directory and the hash-bound fixed-policy checkpoint through
> the supported private input mechanism. Preserve compatibility checks and
> prepare a separate Linux portability procedure. Do not launch gameplay during
> environment installation, run gameplay optimizer updates, purchase compute,
> or merge pull requests. Report which setup checks passed and which runtime
> inputs are still missing before publishing the environment.

## First cloud development task

After publishing, start a task on the migration branch with this handoff:

> Continue the Heroes RL engineering work using the cloud handoff document.
> Review newer commits on codex/builtin-opponent-evaluation and bring the required
> terminal-accounting fixes into this branch. Keep historical Mac failures and
> outcomes intact. Complete Linux portability and a separately recorded short
> native validation for both colors, then prepare the bounded Nullkiller Easy
> comparison using unchanged frozen weights and explicit budget accounting.
> No gameplay training or PPO updates. Commit coherent code/configuration and
> compact evidence, push this explicit branch, and leave pull requests unmerged.
> If blocked on private inputs or runtime permissions, identify the exact missing
> item and continue independent source/testing work. Report what actually ran,
> what passed, and what remains; never label an incomplete setup a working game.

## Debian 13 cloud implementation

The migration branch now includes `9193db35857e07ddbc45e98a1dd7c3f4a8c07da0`.
Its terminal diagnostics remain historical Mac evidence; merging their records
is not a Linux gameplay check.

The current cloud host is Debian 13 amd64 with CPython 3.12.14. It provides
neither root package-install privileges nor `sudo`. The original bootstrap's
`apt` stage therefore fails on this host. The additional tested installation
path uses authenticated Debian metadata and hash-bound package archives under
`/workspace/cloud-setup`, with existing host runtime libraries linked into that
prefix. It does not run package maintainer scripts or alter system directories.
The local prefix remains dependent on the recorded Debian base; the package
archive list is not a standalone operating-system image.

```sh
cd /workspace/heroes-rl
bash tools/setup_cloud_locked.sh
source /workspace/cloud-setup/activate.sh
python -m unittest discover -s tests -v
python tools/prepare_vcmi.py --check
```

`dependencies/learner-linux-x86_64-py312.lock` pins every resolved CPU Python
package and its exact Linux wheel hash. It was also installed independently into
a fresh virtual environment with `--require-hashes` and `pip check`.
`dependencies/cloud-debian13-amd64.json` binds the downloaded Debian archives;
`base-system-packages.tsv` under the local prefix records the host dependencies.
The macOS learner lock and original learner source identities remain unchanged.
Two compiler jobs and one PyTorch thread are the default. Source the activation
file in each new shell; processes and shell exports do not survive publication.
No persistent service or automatic gameplay startup is required.

### Private input inventory

After the user supplies an approved private mount or artifact location, inventory
the selected inputs without copying them into the checkout:

```sh
python tools/inventory_cloud_inputs.py --data /private-inputs/game \
  --checkpoint /private-inputs/fixed-policy.pt \
  --output /private-inputs/manifest.json
```

The command verifies the exact original checkpoint identity and writes relative
Data file names, sizes and SHA-256 hashes to a new mode-0600 manifest outside
the repository. It rejects empty data and nested symlinks/nonregular entries.
This is a byte inventory, not proof of complete/usable game content, legal
ownership, checkpoint portability, or successful gameplay. Both inputs and this
manifest must stay private and outside shared snapshots.

### Explicit frozen-policy portability

The original fixed checkpoint is still an input, never a generated substitute.
The only accepted public migration input has the full SHA-256 and byte count in
**Private inputs** above. The shorter, differing digest in the later chat is not
an alternative accepted identity. Preserve the original in private storage.

```sh
source /workspace/cloud-setup/activate.sh
python tools/port_fixed_policy_linux.py \
  --checkpoint /private-inputs/fixed-policy.pt \
  --output /private-inputs/linux-policy-v1
```

These paths are placeholders for user-approved private storage, not assumed
mounts. The command refuses an existing output. It verifies the original file,
Torch 2.9.1/Python 3.13.5 provenance, v7 contracts, original learner source hashes,
learner and opponent identities, model shapes and finite tensors. It creates a
separate policy identity and runtime provenance for Linux CPU Torch 2.9.1+cpu,
then uses the standard strict loader and exact tensor comparisons for both
models. No compatibility flag is disabled, optimizer restored or update run.
The original checkpoint and historical plans remain untouched. Its private
`portability.json` binds both files and records that cross-platform behavioral
equivalence and Linux gameplay have **not** been established.

Synthetic portability tests cover success, immutable original/output behavior,
wrong file identity, source/runtime drift, contract corruption, frozen-opponent
identity drift and tensor corruption. Those tests do not validate the absent
original checkpoint. The original Mac evaluators intentionally continue to
reject the converted file's different digest. The separate Linux plan below binds the portability report, converted digest,
actual native runtime and original exact model weights. It independently
compares original and loaded learner/opponent tensors, including dtype, shape
and bits. Do not patch the old Mac plans or overwrite their identities.

Private Data, the original checkpoint and any raw games must remain outside
Git and shared snapshots. Until those inputs are supplied and checked, real games and their native replay
remain blocked. Asset-free CLI/library checks are separately available with
`tools/check_cloud_runtime.py`.
Neither installing dependencies nor compiling replay establishes a working game.

### Verified development state and continuation

[The compact cloud record](../validation/cloud-cpu-2026-10-25.json) binds the
implementation, dependency locks, seven native artifacts and retained local
logs. It records a 4-core CPU quota (five CPUs in affinity), a 32 GiB memory
limit, GCC 14.2.0, CMake 3.31.6, Ninja 1.12.1, CPython 3.12.14 and CPU Torch
2.9.1+cpu. Installation uses two compiler jobs and one Torch thread.

The original setup record reports all six requested targets built, 925 tests
(922 passed, three skipped), and 65 native packet/request-accounting checks.
The locked installer also repeated with `ninja: no work to do.` This historical
record remains intact. The current continuation rebuilt the replay target for
`c29e96e`, restored the locked environment and passed **989 tests: 986 passed,
three skipped, zero failures**, including 231 native codec/accounting checks.
The same three external-input skips remain: two need original Data and one
needs retained Mac captures. Four isolated CLI checks and four dynamic library
loads pass without a game. See the new
[hash-bound continuation record](../validation/cloud-linux-integration-2026-10-25.json).

The original sudo-based package stage failed on this host. A relocated-library
configure failure and cached static-SDL2 link failure were also preserved and
resolved by the local dependency installer and explicit shared-SDL2 selection.
No engine recipe or game rule was changed. Local logs are retained under
`/workspace/cloud-setup`; hashes, failures and compiler warnings are in the
compact record. A successful `vcmireplay` usage check is not a game replay.

The remaining external gate is a user-approved private location containing
Data and the exact original checkpoint. The supported input directories were
rechecked during this continuation and remain empty. No actual checkpoint
conversion or Linux game/replay has run. The environment configuration is a
draft; publication and restoration in a fresh cloud task remain separate checks.
The existing Debian package layer and locked Python environment have been
restored in this task, independently of the Mac.

### Fresh Linux diagnostic execution

`tools/linux_validation_plan.py` and
`validation/probes/validate_cloud_linux.py` provide preparation, execution and
receipt-checked reporting. They keep all private artifacts outside the checkout.
Preparation does not launch the engine and refuses uncommitted implementation
changes, wrong checkpoint identity, source/runtime drift, mismatched Data
inventory, changed maps or missing native targets. Bindings include the Python
executable, Torch binaries/inference flags, resolved shared libraries and the
actual native config/Mods resources; canonical recipe agreement is rechecked. The public path has no
synthetic-checkpoint override. Synthetic unit tests use an internal fixture
entry point and are not gameplay evidence.

The fixed schedule declares both seats in each separate scope:

| Scope | Episodes / learner colors | Engine / policy seeds | Ordinary / hard choices | Day limit | Row ceiling |
| --- | --- | --- | --- | --- | --- |
| Townless diagnostic fixture | 1 Red, 2 Blue | 251025301 / 251025302 | 256 / 264 | 12 | 180 s each |
| Ordinary Tavern-enabled map | 3 Red, 4 Blue | 251025401 / 251025402 | 2040 / 2048 | 200 | 1200 s each |

All four rows also share one 3,600-second maximum, including identity checks,
collection, reconstruction, native replay and cleanup. The existing ordinary
choice/day limits are retained. Each row reserves 30 seconds for cleanup and
final identity checks; an independent guardian stops its worker two seconds
before its ceiling, leaving parent cleanup time within the same bound. These
are maximums, not permission to retry or fill unused time with other games.
The first failed or incomplete row stops the remaining schedule; verified
cutoffs may continue. OS scheduling and filesystem latency are measured, not
claimed to be hard real-time. A deadline overrun cannot admit outcomes.

Every action is sampled from the unchanged frozen model. The fixture generator's
historical scripted evaluator declaration is not executed. Nullkiller Easy and
automatic BattleAI are fixed. The native opponent has no fabricated neural
history. A fresh learner collector and recurrent state are created for each
case. Independent reconstruction folds the complete permitted history; coverage
reports distinguish actually observed intervening/transient sightings from
source-level support. A diagnostic with no such sighting cannot prove it occurred.

A natural outcome receives +1/-1 only after independent policy reconstruction,
full native request verification (including terminal-unapplied attempts), final
identity checks and process cleanup. Verified cutoffs and technical failures
remain unscored. Worker results stay pending until those checks finish and a
hash-bound completion receipt is durable. A separate campaign receipt is required
to read admitted results; a worker/campaign crash never upgrades a pending row.
Failures, partial evidence and not-started rows remain in the original directory.
Nothing is eligible for gameplay optimization. A finished schedule containing
only cutoffs does not establish a natural outcome. Missing or modified retained
captures invalidate report admission even when completion receipts remain.

After private inputs arrive, use a new private output directory. These commands
are a continuation recipe; the paths are placeholders for approved private
storage and the real-input steps have not run here:

```sh
source /workspace/cloud-setup/activate.sh
cd /workspace/heroes-rl
python tools/check_cloud_runtime.py --output /private-inputs/runtime-check.json
python tools/inventory_cloud_inputs.py --data /private-inputs/game \
  --checkpoint /private-inputs/fixed-policy.pt \
  --output /private-inputs/input-manifest.json
python tools/port_fixed_policy_linux.py \
  --checkpoint /private-inputs/fixed-policy.pt --output /private-inputs/linux-policy-v1
python -m validation.probes.create_boundary_fixture --kind natural-townless \
  --output /private-inputs/cloud-fixture-v1
python tools/linux_validation_plan.py \
  --original-checkpoint /private-inputs/fixed-policy.pt \
  --checkpoint /private-inputs/linux-policy-v1/fixed-policy-linux.pt \
  --portability /private-inputs/linux-policy-v1/portability.json \
  --input-manifest /private-inputs/input-manifest.json --data /private-inputs/game \
  --binary vcmi/build/bin/vcmiclient --replay-binary vcmi/build/bin/vcmireplay \
  --map experiments/explorer/maps/hero-management-v1.vmap \
  --fixture-map /private-inputs/cloud-fixture-v1/natural-townless.vmap \
  --output /private-inputs/cloud-diagnostic-v1
# Use the exact plan SHA-256 printed by preparation, not a saved Mac plan hash.
python -m validation.probes.validate_cloud_linux execute \
  --plan /private-inputs/cloud-diagnostic-v1/plan.json --sha256 PRINTED_PLAN_SHA256
python -m validation.probes.validate_cloud_linux report \
  --plan /private-inputs/cloud-diagnostic-v1/plan.json --sha256 PRINTED_PLAN_SHA256
```

Do not run installation or publish an environment snapshot with these private
inputs bundled into it. The reusable runtime check uses a disposable XDG profile:
direct VCMI version checks can otherwise fail while creating a profile under the
cloud user's read-only home. A replay usage check is still not a game replay.
