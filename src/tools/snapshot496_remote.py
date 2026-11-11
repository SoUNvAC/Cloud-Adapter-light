from pathlib import Path
import re,json
r=Path('/home/scv/shared/phase496_20261110')
for v in ['control','texture']:
 s=(r/(v+'.log')).read_text(errors='replace')
 rows=re.findall(r'\|\s*(?:surface_visible|cloud|shadow)\s*\|[^\n]+',s)
 print(v,rows[-3:])
 print((r/v/'UPDATE_CHECK.json').read_text())

