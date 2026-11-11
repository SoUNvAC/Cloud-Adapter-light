from pathlib import Path
import shutil,json
r=Path('/home/scv/shared/phase491_20261108').resolve();assert r==Path('/mnt/d/CloudAdapterExperiments/phase491_20261108')
assert all((r/f'{a}_EXIT_CODE').read_text().strip()=='1' for a in ['A','B'])
assert all(not (r/a/'network_run/STATUS.json').exists() and not (r/a/'network_run/branch_0.pth').exists() for a in ['A','B'])
d=r/'startup_failure_001';d.mkdir(exist_ok=False);(d/'code').mkdir()
for p in list(r.glob('*.py'))+list(r.glob('*.sh'))+[r/'INITIAL_CODE_SHA.json']:
 shutil.copy2(p,d/'code'/p.name)
for name in ['A','B','A.log','B.log','A_EXIT_CODE','B_EXIT_CODE','finish.log','POSTPROCESS_FAILURE.json']:
 p=r/name
 assert p.resolve().is_relative_to(r)
 if p.exists():shutil.move(str(p),str(d/name))
(d/'README.json').write_text(json.dumps(dict(reason='Phase number replacement changed dataset length assertion 8490 to 8491 before training; no step or checkpoint executed.',training_steps=0,preserved=True),indent=2))
print('STARTUP_FAILURE_ARCHIVED_NO_TRAINING')
