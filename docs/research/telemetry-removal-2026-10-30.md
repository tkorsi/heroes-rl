# Removal of training performance telemetry, 30 October 2026

## Prospective protocol

The user requests removal of the CPU instrumentation subsystem. This change
removes native active-worker statistics, phase clocks, counters and Python phase
profiling. It retains parallel execution, numeric ordering, bounded queues,
deadlines, ownership, replay/admission checks, checkpoint preservation and the
manual launcher's total elapsed-time output. The cost of the removed profiling
has not been isolated; no speedup is claimed.

The completed manual session continued checkpoint 1585 to 1728 in 143 complete
updates, admitting 572 losses. Its 24 evaluation games were excluded from PPO.
The original target has 604 games remaining and sequence 1369 is next. The
canonical and backup full state restore exactly. These files, the original
campaign ledger and all historical raw evidence remain unchanged.

Validation uses a clean native build in a new private directory, exact numerical
and failure-cleanup controls, a complete copied update, fresh native controls,
and frozen copied checkpoint1728 policy games. Every diagnostic has bounded
deadlines, separate outputs and verified cleanup. No diagnostic experience is
admitted to official training. A new acceptance descriptor must pass the existing
source, engine, replay and checkpoint gates before selecting the new build.

The user explicitly chose explanation only for self-play. Nullkiller Easy,
the map, policy architecture, PPO configuration, four-game batches, terminal-only
rewards and revealed-movement support remain unchanged. Self-play is a separate
possible experiment; the current v8 continuation cannot switch opponent names
without a new validated provenance and admission contract.

## Results

Pending completion of the declared validation. Official training is not launched.
