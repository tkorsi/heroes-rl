# Heroes experiment workspace

Read `README.md` before changing this project. It is the single authoritative
document for the purpose, agreed constraints, experiment stages, and current
status. Update it when an experiment changes these decisions.

This chat/workspace is the Heroes experiment track. RL lessons and Breakout
belong to the separate learning chat. Do not start a training campaign as part
of a recorder check, or describe built-in AI footage as a learned policy.

This is one Git project. All authored code, engine patches, dependency locks and
experiment configuration must be tracked in the Heroes root so SourceTree shows
every change. `vcmi/` is an ignored generated dependency checkout, not a project
submodule. The canonical engine modifications are `engine/patches/*.patch` and
`engine/ExplorerRL/`. Never leave new work only in the dependency checkout: edit
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
