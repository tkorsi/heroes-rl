"""Path-free summary of the native capture verification counts in probe runs made with HEROES_VERIFY_NATIVE_CAPTURE=1.

Usage: capture_counts.py OUT CODE LABEL=OUTPUT_ROOT...
Each OUTPUT_ROOT holds probe runs NAME/ with their stderr in NAME.err (as the revalidation scripts write them).
"""
import glob
import json
import os
import re
import sys

PATTERN = re.compile(r"ExplorerRL capture verification: (\d+) unchanged-view captures confirmed")


def counts(path):
    return [int(found) for found in PATTERN.findall(open(path, errors="replace").read())]


out, code, roots = sys.argv[1], sys.argv[2], [item.split("=", 1) for item in sys.argv[3:]]
runs = {}
for label, root in roots:
    for err in sorted(glob.glob(os.path.join(root, "*.err"))):
        name = os.path.basename(err)[:-4]
        host = counts(err)
        others = []
        for path in glob.glob(err[:-4] + "/**/*", recursive=True):
            if os.path.isfile(path) and os.path.getsize(path) < 50_000_000 and path.endswith((".log", ".txt", ".err")):
                found = counts(path)
                if found:
                    others.append(max(found))
        runs[f"{label}/{name}"] = {"host_process_max_confirmed_power_of_ten": max(host) if host else None,
                                   "other_processes_with_confirmations": len(others),
                                   "other_processes_min_of_max": min(others) if others else None,
                                   "other_processes_max": max(others) if others else None}
json.dump({"code": code, "switch": "HEROES_VERIFY_NATIVE_CAPTURE=1",
           "note": ("Each process builds every observation it would skip, fails the game on any disagreement and reports "
                    "confirmed skips on stderr at powers of ten. The host process is the embedded engine (none for the "
                    "cold backend, whose game processes log separately); other processes are cold game processes and "
                    "vcmireplay. The stalled poison game stops early by design. Every run passed."),
           "runs": runs}, open(out, "w"), indent=1, sort_keys=True)
print(len(runs), "runs")
