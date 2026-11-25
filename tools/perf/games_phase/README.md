# Games-phase measurements of 2026-11-25

The tools and queue scripts behind the journal entries of 2026-11-25 in
`docs/research-log.md` ("The Games phase ..." to "Handoff: the Games phase of a
100-game batch"). They are kept here as they ran: they read and write the work
area `~/Projects/Heroes-artifacts/claude-g10` (frozen source trees `src-*`, a copy
of checkpoint 2036 in `base/`, state folders `states/`, logs, results and recorded
replays), and the shell scripts start real engines and the normal training
command. Run them only on a machine with nothing else training, each with its own
state folder.

| File | What it does |
| --- | --- |
| `full-run.sh`, `full-run2.sh` | One timed normal batch (`./train.command 100`) on a fresh copy of checkpoint 2036, under `/usr/bin/time -l`, with `sampler.sh` writing the processor use of actors and engines every 5 s; `full-run2.sh` also logs `PYTHONPATH`. |
| `batch_report.py` | The numbers of a batch from the trainer's own log, and a game-by-game comparison with another batch on the same seeds. |
| `verify_train.py`, `verify-pair.sh`, `verify-one.sh`, `verify-one2.sh` | The normal command with the batch put in game-number order before PPO and a digest of every trajectory and of PPO's input: two trees on the same seeded games give the same lines when they play and train alike. |
| `actor_replay.py`, `ab.sh` | Record a live game's engine messages; replay them through the actor alone (one torch thread, no cyclic collector, as an actor), checking every answer and the trajectory digest; `ab.sh` alternates two trees. |
| `profile_one.py`, `live-ab.sh`, `phases.py`, `permits_count.py` | One live game with actor and engine processor time and the trajectory digest; profiling helpers. |
| `window_bench.py` | Keep N actors busy for a window and count decisions per second. |
| `engine-ab.sh`, `build-e2.sh`, `build-engine.sh` | Seeded games on two engine runtimes (digests, days, choices); build and freeze an engine runtime. |
| `snap.sh`, `when-quiet.sh` | Freeze the working tree as `src-NAME`; wait until no training command runs. |
| `queues/queue-*.sh` | The sequences launched overnight through `claude-speed/bin/run-normal-priority.sh`, in order. |
