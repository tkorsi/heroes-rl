# Curriculum v1 research record

The append-only [event log](../../validation/curriculum-v1-research-log.jsonl) records preregistration, unsuccessful candidates, validation and results.

Question: can changing only initial game states give unchanged update 146 both natural wins and losses under its own stochastic actions? No scripted rush trajectories are used. Stage 0 is a townless capture opportunity; farther/weaker starts and the original map are later hypotheses, not demonstrated curriculum achievements.

The configuration predeclares separate diagnostic and held-out seeds. Seat results will remain separate. Preparation does not consume or start the new two-hour campaign window. Native validation and do-nothing controls must pass before PPO. The outcome criterion is held-out win-rate change, not weight movement.

## Measured selection result

The bounded study completed 32 native games with unchanged update 146: 8 on the
original 20×20 townless candidate and 24 on the open 8×8 candidate. Each candidate
was tested in both seats. All 32 were natural losses on day 7, with 2490 sampled
choices and zero admitted training samples or optimizer steps. Native replay,
terminal/history validation and exact policy reconstruction passed for every
game. The 20×20 games sampled 80 moves and the 8×8 games 144; none targeted the
opponent town entrance. These are measured observations, not proof that victory
is unreachable. Sparse coordinate exploration is a plausible explanation.

The mixed-outcome gate failed, so no PPO campaign/session was created and no
new two-hour budget began. Do-nothing controls and held-out evaluation were not
run: selection already failed, and there is no trained checkpoint to compare.
No automatic-win or learned-strategy claim is made. Four scenario/seed tests
passed. Owned native cleanup and exact update 146 restoration were rechecked.

The two initial loader errors occurred before gameplay and are logged explicitly;
the canonical checkpoint metadata binding was corrected before the 32 games.
Stage 1 (farther/weaker start) and return to original difficulty remain untested
hypotheses. No failed diagnostic, previous scripted rush sequence or evaluation
was admitted to PPO. See [compact results](../../validation/curriculum-v1-selection-2026-10-26.json).
