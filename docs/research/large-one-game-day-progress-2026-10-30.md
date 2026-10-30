# Trained one-game continuation with calendar progress

The user stopped small-network experiments and requested another manually launched
game, then requested the current day in its progress bar. The selected model is
the trained checkpoint 1753 (64 hidden units, 32 features, 74,344 parameters).
The tiny initialization, completed capture and consumed seeds remain preserved.
No model is resized, reinitialized, retrained or replaced in checkpoint storage.

```sh
cd /Users/orsi/Projects/Heroes
./play-one-selfplay.sh --check
./play-one-selfplay.sh --run
```

The new driver reuses the existing large-model seed ledger, so the first attempt
continues at its next unused offset rather than replaying a seed. Check mode
restores and validates the complete published model/Adam/RNG/progress/pool state,
loads two independent frozen policy copies and verifies the accepted native runtime.
Only explicit `--run` reserves a new seed and starts one physical game. There is
no optimizer update or training admission. The map, automatic combat, terminal-only
rewards, 9,920 ordinary plus 320 reserved choices per player, day-365 horizon,
1,200-second capture deadline and 3,600-second default total deadline remain.

The requested troop-slot restriction, explicit garrison hero entry/exit and
`own-army-details-v1` remain selected. This is an explicit inference experiment
using the trained policy under the new support/observation contract, not a rewrite
of the original checkpoint metadata. The old 1753 evaluator retains its historical
mode. The existing runtime validator is reused without loading tiny weights or
using the tiny seed ledger.

## Calendar display

The collector delivers only already-observed player/day/limit values to an optional
UI hook. The disposable progress snapshot stores each seat's latest observed day;
the heading displays the shared calendar's latest known day, for example
`Game slot 1 | Day 42/365 | collecting`. Before any day is delivered it shows `?`.
Both Red/Blue action bars and the final running-time output remain. Day 366 may be
shown during finalization of the configured day-365 cutoff; the display does not
invent or clamp observations. No action is sampled after the existing cutoff.

There is no extra history scan, engine polling or write frequency increase. The
small snapshot retains its once-per-second write throttle. Old snapshots remain
readable; malformed calendar metadata and observer failures cannot veto gameplay.
The display never controls action selection, RNG, rewards, replay or admission.

## Validation and provenance

The native runtime and original native-acceptance receipt are unchanged. The
runtime verifier accepts an explicitly reviewed display-only source delta for
three exact files: the UI, its collector callback, and historical source-pin
metadata. Every other accepted source remains exact. The proof labels this as a
reviewed display transition, not a newly performed native capture. Altered,
missing or extra source bytes still fail closed. Canonical engine agreement and
the real clean-checkout shell preflight are release checks.

Offline checks used hash-bound first-decision histories for both seats from the
already accepted passive-army native prefix. The actual trained 1753 model had
finite action/value outputs and no blocked action kinds. Its C++ log probability,
value, recurrent state and conditional probabilities matched Python exactly;
C++ prefix execution with one and four workers also matched. Model weights,
parameter storage independence, Torch RNG and empty gradients were preserved.
No optimizer, backward pass, game, native replay or seed reservation was performed.
This is shape/numerical integration coverage, not a full large-model game under
the new restriction or evidence of playing strength.

Two offline harness attempts failed before this successful result: a stdin syntax
typo and the native compiler's existing rejection of frozen parameter flags. The
successful control enabled flags on disposable loaded copies inside `no_grad`
and restored them, without optimizer creation or updates. A launcher test also
initially used a temporary path outside the production artifact root; the fixture
was corrected to mock and assert that boundary, leaving the production check intact.

Final focused tests, exact preservation and shell readiness are recorded in the
compact validation receipt after checking. No training or full game is launched
as part of preparation.

The final focused suite ran 124 tests: 123 passed and one was skipped. Shell
syntax and canonical VCMI agreement passed. Full-suite and native gameplay
were not repeated for this launcher/display change.
