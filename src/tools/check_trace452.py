from collect_phase241 import ssh
print(ssh('''from pathlib import Path
import json
r=Path('/home/scv/shared/phase452_20261106');a={x['step']:x for x in map(json.loads,(r/'A/network_run/INPUT_TRACE.jsonl').read_text().splitlines())};b={x['step']:x for x in map(json.loads,(r/'B/network_run/INPUT_TRACE.jsonl').read_text().splitlines())};common=sorted(a.keys()&b.keys());assert all(a[k]==b[k] for k in common);print(json.dumps(dict(batch_trace_equal=True,checked_steps=common)))
''').decode())
