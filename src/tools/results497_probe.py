from pathlib import Path
import json
r=Path('/home/scv/shared/phase497_20261110')
for row in json.loads((r/'RESULT.json').read_text()):print(row['checkpoint'],row['metrics'])
print((r/'RESUME_TRACE_CHECK.jsonl').read_text())
