from pathlib import Path
import re
r=Path('/home/scv/shared/phase497_20261110')
for n in ['RESUME_TRACE_CHECK.jsonl','texture/UPDATE_CHECK.json']:
 print(n,(r/n).read_text())
s=(r/'resume.log').read_text(errors='replace')
print('\n'.join(re.findall(r'\|\s*(?:surface_visible|cloud|shadow)\s*\|[^\n]+',s)[-3:]))
print(s[-700:])
