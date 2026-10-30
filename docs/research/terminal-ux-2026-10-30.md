# Terminal presentation update

The user requested clearer grouping, readability and colors for the chosen
settings and restrictions. This update changes presentation only. It preserves
checkpoint 1753, the full model/optimizer/RNG/progress/sampler/pool state, map,
policy support, passive information, limits, rewards and manual launch boundary.

An active user game was detected before any edit. All work was performed in the
separate `Heroes-ui-preview` checkout while its main checkout remained clean at
`c83f9027b94a4bca83fcaf5a31618a0440a7d0a6`. Synchronization waited for that
run's verified shutdown and held the campaign lease to prevent a new-run race.
No game, replay, restart, signal or seed reservation is part of this UI task.

## Presentation

- Terminal startup/check output groups Setup, Game rules and Limits. It explicitly
  identifies the trained large model, frozen evaluation, allowed recruitment and
  whole-hero garrison changes, disabled stack operations/upgrades, automatic combat,
  revealed-only movement and passive own-army information.
- Green marks readiness/allowed actions, amber marks temporary restrictions,
  cyan marks headings, and Red/Blue use matching player colors. Labels retain
  their meaning without color.
- Live progress separates day/phase, player action bars and mandatory-dialog
  reserves. Bars represent choices consumed, never percentage of game completion.
- Result summaries distinguish verified recording from a natural win or unscored
  cutoff. Terminal errors retain a nonzero exit and do not retry automatically.
- `NO_COLOR`, `TERM=dumb`, redirected output and narrow terminals have readable
  fallbacks. No third-party terminal package was added.
- Default non-terminal output preserves its historical JSON-plus-timing contract.
  Explicit `--json` prints only JSON on stdout and timing on stderr.

The progress snapshot schema, calendar callback, polling cadence and once-per-second
snapshot write throttle are unchanged. Native runtime evidence is inherited via
the existing exact source transition; only the UI source and its source-pin metadata
change within that reviewed inventory. The native receipt is not rewritten and no
new native gameplay is claimed.

## Verification

The focused suite ran 132 tests, with 131 passes and one skip. It covers TTY and
machine-readable output, color opt-outs, narrow wrapping, failure exit semantics,
unchanged collector/RNG behavior, source gates and launcher boundaries. A later
label-spacing-only pass ran 13 launcher tests successfully. Visual review used a
synthetic ANSI-to-HTML preview of the actual render functions; colors depend on
the user's terminal theme. This is a UI preview, not gameplay footage.

The actual main-checkout `--check --json` passed after the prior user run
shut down, with the campaign lease held throughout synchronization. Both actual
PTY checks passed (80-column color and 44-column NO_COLOR). Canonical engine
agreement passed, all 137 protected files are unchanged, and no seed was
reserved. See [compact validation](../../validation/terminal-ux-2026-10-30.json).
No new native capture or full-suite run was performed for this presentation change.
