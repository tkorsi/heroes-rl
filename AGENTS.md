# Heroes experiment workspace

## Rule 1: the user launches scripts and training

By default, the user personally launches project scripts and all training
outside Codex. Prepare the code, script and exact command, then hand them to the
user. Do not start or resume training, self-play, evaluation games, replay jobs,
profiling, numerical probes or benchmarks yourself. This includes PPO updates
on saved games and isolated CPU/Metal experiments without checkpoint publication.

An exception requires an explicit user instruction authorizing assistant
execution and applies only within that instruction's scope.
Requests to fix, optimize, investigate or continue work are not launch permission;
permissions for earlier runs do not authorize later ones. This applies equally
to direct tool calls, inline code, detached/background processes, supervisors, subagents,
remote jobs and scheduled tasks.

Code/file/Git work and lightweight read-only inspection remain allowed. When
asked to check progress, inspect the user's existing process, logs, metrics or
status endpoint without starting or restarting a workload. Existing scripts may
expose lightweight status information for this purpose. Do not keep a Codex turn
alive with unattended waiting or repeated polling by default; continuous
monitoring requires an explicit request. User-launched work must run independently
of the Codex conversation. This rule does not waive runtime or validation gates.

## One main development line

Do not merge branches/PRs or push directly to `main` without the user's explicit
instruction for that publication. Preparing changes or recording research does
not authorize publication. When publication is desirable, ask and wait. An
explicitly requested experiment may use its own local branch; do not merge it
back or retire it automatically.

`main` is the canonical development branch and `/Users/orsi/Projects/Heroes` is
the working checkout. Keep current implementation, configuration and research
findings together there; do not accumulate long-lived experiment branches or
duplicate working directories. Publish incremental, focused commits to `main`
when authorized. Reintegrate and remove temporary isolation only after the user
explicitly authorizes that step. Preserve required historical source identities, checkpoints
and runtime assets in private storage before retiring a working copy.

Read `README.md` before changing this project. It is the single authoritative
document for the purpose, agreed constraints, experiment stages, and current
status. Update it when an experiment changes these decisions.

## Research journal

Keep a chronological research journal in `docs/research-log.md` for every
meaningful step in this project, including investigation, design decisions,
implementation, validation, manual launches, failures, corrections and results.
Record the question or hypothesis and acceptance criteria before an experiment;
afterward record what actually ran and what the evidence supports. This journal
must be usable when preparing a research paper, not only for successful runs.

Include the source/runtime/configuration identity, parent checkpoint, seed
allocation, changed settings, commands or durable evidence references, timing,
denominators, validation status and limitations when relevant. Separate intended,
implemented, tested and observed behavior. Distinguish game outcomes from
verification failures, training admission and checkpoint publication. Preserve
earlier entries; append explicit corrections when a prior claim was incomplete
or wrong. Never infer learning solely from more decisive self-play games.

Journal entries and committed documentation must be in English. Keep raw game
captures, checkpoints, private data and bulky diagnostic artifacts in private
storage; reference them rather than copying them into the public repository.
Summarize routine repetitive operations, but record every experiment attempt,
including failed or interrupted attempts. Journaling does not authorize a game,
training launch, publication, or changes outside the user's requested scope.

This chat/workspace is the Heroes experiment track. RL lessons and Breakout
belong to the separate learning chat. Do not start a training campaign as part
of a recorder check, or describe built-in AI footage as a learned policy.

This is one Git project. All authored code, engine patches, dependency locks and
experiment configuration must be tracked in the Heroes root so SourceTree shows
every change. `vcmi/` is an ignored generated dependency checkout, not a project
submodule. The canonical engine modifications are `engine/patches/*.patch`,
`engine/ExplorerRL/`, `engine/ExperimentReplay/`, and `engine/NativeSession/`.
Never leave new work only in the dependency checkout: edit
the canonical files or export a changed source file to the corresponding patch,
then run `python3 tools/prepare_vcmi.py --check` to verify exact agreement.
Follow VCMI's own AGENTS.md when changing its source. Keep upstream source,
copyrighted game data, builds, raw frames, videos and large checkpoints out of
Git. Commit compact validation results instead.

Keep committed documentation, comments and GitHub metadata in English. This
repository is Heroes RL only; unrelated learning projects and game installations
remain local and ignored. Publish explicit branches to GitHub, never mirror all
refs: local editor checkpoint refs may contain unrelated historical files.

For ESM JavaScript from stdin, use `node --input-type=module - <<'NODE'`.
With arguments, use `node --input-type=module - "$arg1" <<'NODE'`.
Plain `node -` must use CommonJS syntax only.
