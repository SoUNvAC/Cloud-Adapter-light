from pathlib import Path
import re
r=Path('/home/scv/shared/phase497_20261110')
s=(r/'texture.log').read_text(errors='replace')
print('\n'.join(re.findall(r'\|\s*(?:surface_visible|cloud|shadow)\s*\|[^\n]+',s)[-3:]))
