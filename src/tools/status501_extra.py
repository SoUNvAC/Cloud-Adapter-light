from pathlib import Path
r=Path('/home/scv/shared/phase501_20261111')
for n in ['postprocess.log','POSTPROCESS.json','UNIT_ACCEPTANCE.json','SMOKE_PARAMETERS.json']:
 p=r/n
 print(n,p.read_text()[-1500:] if p.exists() else 'MISSING')
