from pathlib import Path
T=Path(__file__).parent
def put(n,s):(T/n).write_bytes(s.encode('utf-8'))
base='/home/scv/shared/phase483_20261108'
for n in ['train_shadow482.py','infer_shadow482.py','verify_shadow482.py','ablate_shadow482.py','boundary_shadow482.py','boundary_effect482.py','verify_archive482.py']:
 s=(T/n).read_text(encoding='utf-8').replace('482','483')
 if n=='train_shadow482.py':
  s=s.replace("ARM=sys.argv[1];assert ARM in ['A','B']", "ARM=sys.argv[1];assert ARM=='B'\nSEED=int(sys.argv[2]);assert SEED in [45202,45203]\nINIT_SEED=48000+SEED-45200")
  s=s.replace("ROOT=Path('"+base+"')", "ROOT=Path('"+base+"')/f'seed_{SEED}'")
  s=s.replace('45201','SEED').replace('torch.manual_seed(48001)','torch.manual_seed(INIT_SEED)').replace('init_seed=48001','init_seed=INIT_SEED')
 else:
  s=s.replace("ARM=sys.argv[1];R=Path('"+base+"')/ARM/'network_run'", "ARM=sys.argv[1];SEED=int(sys.argv[2]);R=Path('"+base+"')/f'seed_{SEED}'/ARM/'network_run'")
  s=s.replace("R=Path('"+base+"')/sys.argv[1]/'network_run'", "R=Path('"+base+"')/f'seed_{int(sys.argv[2])}'/sys.argv[1]/'network_run'")
 put(n.replace('482','483'),s)
put('shadow_redistribution483.py',(T/'shadow_redistribution482.py').read_text(encoding='utf-8'))
put('run_shadow483.sh','''#!/bin/bash
set -eu
cd /home/scv/shared/phase483_20261108
SEED="$1"
PY=/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python
trap 'code=$?; echo "$code" > "seed_${SEED}_EXIT_CODE"' EXIT
$PY train_shadow483.py B "$SEED"
$PY verify_shadow483.py B "$SEED"
$PY boundary_shadow483.py B "$SEED"
$PY ablate_shadow483.py B "$SEED"
$PY boundary_effect483.py B "$SEED"
touch "seed_${SEED}/B/DONE"
''')
