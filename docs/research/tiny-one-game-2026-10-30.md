# Manual one-game tiny-policy evaluation

The user requested a script they will launch themselves, using the small model.
The earlier request to run a game was interrupted before any game or seed
reservation started. This preparation launches no gameplay or training.

```sh
cd /Users/orsi/Projects/Heroes
./play-one-tiny-selfplay.sh --check
./play-one-tiny-selfplay.sh --run
```

The script checks by default. Only explicit `--run` starts one physical self-play
game. Both seats use independent copies of the preserved **untrained**
`tiny-v1-untrained-initial` policy: hidden size 8, feature size 4, 15,460 parameters.
The weights were randomly initialized, without resizing, transferring or distilling
checkpoint 1753. The original trained model and its complete Adam/RNG/progress/pool
state remain a protected comparison baseline, not the tiny model's parent.

The existing map, v8 action interface, revealed-target and visible-interaction
budget support remain unchanged. Both players receive 9,920 ordinary choices and
320 reserved mandatory-dialog choices, with a 10,240 hard ceiling and day 365
horizon. The script retains the existing 1,200-second capture deadline and a
3,600-second total deadline by default; `--seconds` controls only the total safety
deadline. It prints total elapsed time on success, failure or handled interruption.

The visible Red/Blue choice bars, compact external-history storage, full two-policy
reconstruction, native replay, owned-process cleanup and original-state protection
are shared with the accepted one-game evaluator. Narrow explicit callbacks select
the tiny checkpoint and its own identity checker; the existing 1753 script keeps
its original defaults. The shared supervisor executes this new driver's worker,
which cannot silently fall back to the 1753 loader.

Private outputs use `Heroes-artifacts/selfplay-tiny-one-game`. A separate locked
ledger consumes each attempt before launch, including failed or unstarted attempts.
Its engine seed namespace begins at 1,974,000,000 and policy seeds at 1,976,000,000,
separate from the original campaigns, 1753 evaluation and tiny initialization.
Missing or altered reservation evidence fails closed. Check mode does not create
the output namespace, acquire a campaign lease, reserve seeds or launch processes.

The loader pins the prepared policy/receipt bytes and verifies exact metadata,
random-initialization provenance, model shape, parameter count, weight digest and
initial frozen pool. Each seat owns separate parameter tensors, recurrent state
and sampling RNG. All gameplay is diagnostic-only: no Adam state is created,
no PPO sample is admitted, and no weight update occurs. Four-game training batches
are unchanged; `training_launcher_accepted` remains false.

Offline preparation cannot establish successful native tiny-model gameplay or
playing strength. The user's first manual game will provide that evidence; failures
must remain recorded and must not trigger an automatic retry.
