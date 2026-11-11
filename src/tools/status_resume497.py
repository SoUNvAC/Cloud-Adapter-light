from pathlib import Path
r=Path('/home/scv/shared/phase497_20261110')
for name in ['STATUS.json','RESUME_ACCEPTANCE.json','RESUME_FAILURE.txt']:
 p=r/name
 if p.exists():print(name,p.read_text())
p=r/'resume.log'
if p.exists():print(p.read_text(errors='replace')[-1800:])
