# Heroes experiment workspace

Read `README.md` before changing this project. It is the single authoritative
document for the purpose, agreed constraints, experiment stages, and current
status. Update it when an experiment changes these decisions.

This chat/workspace is the Heroes experiment track. RL lessons and Breakout
belong to the separate learning chat. Do not start a training campaign as part
of a recorder check, or describe built-in AI footage as a learned policy.

Keep project code and experiment configurations in Git. VCMI is a nested Git
submodule; follow its own AGENTS.md and commit its changes before updating the
parent pointer. Keep copyrighted game data, builds, raw frames, videos, and
large model checkpoints out of Git. Commit compact validation results instead.

For ESM JavaScript from stdin, use `node --input-type=module - <<'NODE'`.
With arguments, use `node --input-type=module - "$arg1" <<'NODE'`.
Plain `node -` must use CommonJS syntax only.
