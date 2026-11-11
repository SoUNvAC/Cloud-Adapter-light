from pathlib import Path
r=Path('/home/scv/shared/phase500_20261111')
print('cached',len(list((r/'predictions496').glob('*.npz'))))
p=r/'DIAGNOSIS496.json'
print(p.read_text() if p.exists() else 'diagnosis not complete')
