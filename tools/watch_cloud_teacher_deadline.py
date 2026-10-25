"""Independent private stop receipt for the explicitly bounded cloud session."""
from pathlib import Path
import datetime,time,json,os,subprocess
root=Path('/workspace/Heroes-artifacts/session');root.mkdir(parents=True,exist_ok=True)
hard=datetime.datetime(2026,10,25,17,53,tzinfo=datetime.timezone.utc).timestamp()
compute=hard-180
identity={'schema':'heroes-cloud-session-stop-v1','watchdog_pid':os.getpid(),
 'started_unix':time.time(),'hard_deadline_unix':hard,'compute_deadline_unix':compute,
 'hard_deadline_utc':'2026-10-25T17:53:00Z','compute_deadline_utc':'2026-10-25T17:50:00Z',
 'mechanism':'Every continuation worker/native child has its own absolute process guardian; this independent watcher records the stop boundary.',
 'state':'armed'}
(root/'deadline-watchdog.json').write_text(json.dumps(identity,indent=2)+'\n')
while time.time()<compute: time.sleep(min(20,compute-time.time()))
(root/'stop-launching.json').write_text(json.dumps({'stop_launching':True,'at_unix':time.time(),'reason':'absolute compute deadline; preservation/publication reserve'},indent=2)+'\n')
while time.time()<hard: time.sleep(min(10,hard-time.time()))
identity.update(state='stopped_deadline',stopped_unix=time.time(),restart_authorized=False)
(root/'deadline-watchdog.json').write_text(json.dumps(identity,indent=2)+'\n')
