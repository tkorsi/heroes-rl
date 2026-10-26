"""Compose reports/cloud/claude-original5000-status.json from live evidence (watchdog, owned processes, notes).

    status.py OUTPUT_JSON        # notes come from /home/claude/overnight/status-notes.json
"""
import json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
HOME = Path('/home/claude/overnight')
sys.path.insert(0, str(HOME))
import watchdog as wd


def main(output):
    state = json.loads((HOME/'watchdog.json').read_text())
    beat = json.loads((HOME/'watchdog-heartbeat.json').read_text())
    alive = wd.identity(state['pid'])
    notes = json.loads((HOME/'status-notes.json').read_text())
    owned = []
    for root in wd.owned_roots():
        tree = [wd.identity(p) for p in wd.descendants(root['pid'])] if root['alive'] else []
        owned.append(dict(label=root['label'], pid=root['pid'], starttime=root['starttime'], alive=root['alive'],
                          command=root['command'][:6], registered_utc=root['registered_utc'],
                          live_descendants=sum(1 for t in tree if t)))
    value = dict(schema='claude-original5000-status-v1',
                 generated_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                 assignment='docs/operations/claude-original5000-overnight-20261026.txt (main b4fe70c)',
                 deadlines=dict(heavy_work_stop_utc=state['heavy_stop_utc'], hard_stop_utc=state['hard_stop_utc'],
                                heavy_work_stop_monotonic=state['heavy_stop_monotonic'], hard_stop_monotonic=state['hard_stop_monotonic'],
                                recorded_utc=state['created_utc'], boot_id=state['boot_id'], immutable=True),
                 watchdog=dict(pid=state['pid'], kernel_starttime=state['starttime'], command=state['command'],
                               alive=bool(alive and alive['starttime'] == state['starttime']),
                               heartbeat_utc=beat['utc'], seconds_to_heavy_stop=beat.get('seconds_to_heavy_stop'),
                               heavy_stop_receipt=state.get('heavy_stop_receipt'), hard_stop_receipt=state.get('hard_stop_receipt'),
                               restarts=state.get('restarts', [])),
                 owned_processes=owned, **notes)
    Path(output).write_text(json.dumps(value, indent=1) + '\n')


if __name__ == '__main__':
    main(sys.argv[1])
