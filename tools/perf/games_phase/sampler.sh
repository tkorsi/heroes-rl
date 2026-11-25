#!/bin/sh
# Every 5 s: time, the CPU of vcmiclient processes and of python actor processes (percent of one core, summed),
# their counts, and the load average. Usage: sampler.sh OUT
while true; do
  ps -A -o pcpu=,comm= | awk -v t="$(date +%s)" '
    /vcmiclient/ {e+=$1; ne++} /multiprocessing|python/ {p+=$1; np++}
    END {printf "%s engine %.0f %d python %.0f %d\n", t, e, ne, p, np}' >> $1
  sleep 5
done
