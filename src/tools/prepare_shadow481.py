from pathlib import Path
T=Path(__file__).parent
def put(n,s):(T/n).write_bytes(s.encode('utf-8'))
base="/home/scv/shared/phase481_20261108"
def convert(s,old):return s.replace(str(old),'481').replace('phase481_20261107','phase481_20261108')
for old,name in [(480,'train_shadow481.py'),(477,'train_control481.py')]:
 s=convert((T/f'train_shadow{old}.py').read_text(encoding='utf-8'),old)
 arm="ARM=sys.argv[1];assert ARM in ['A','B']" if old==480 else "ARM=sys.argv[1];assert ARM=='A'"
 assert arm in s
 s=s.replace(arm,"ARM=sys.argv[1];assert ARM in "+("['A','B']" if old==480 else "['C']")+"\nSEED=int(sys.argv[2]);assert SEED in [45202,45203]\nINIT_SEED=48000+SEED-45200")
 s=s.replace("ROOT=Path('"+base+"')","ROOT=Path('"+base+"')/f'seed_{SEED}'")
 s=s.replace('45201','SEED')
 if old==480:
  # 48001 was renamed to48101 by the phase conversion: restore paired init seeds.
  s=s.replace('torch.manual_seed(48101)','torch.manual_seed(INIT_SEED)')
  s=s.replace("new_module=dict(","new_module=dict(init_seed=INIT_SEED,")
 put(name,s)
put('shadow_redistribution481.py',(T/'shadow_redistribution480.py').read_text(encoding='utf-8'))
put('infer_shadow481.py',convert((T/'infer_shadow480.py').read_text(encoding='utf-8'),480))
for n in ['verify_shadow480.py','ablate_shadow480.py','boundary_shadow480.py']:
 s=convert((T/n).read_text(encoding='utf-8'),480)
 s=s.replace("ARM=sys.argv[1];R=Path('"+base+"')/ARM/'network_run'","ARM=sys.argv[1];SEED=int(sys.argv[2]);R=Path('"+base+"')/f'seed_{SEED}'/ARM/'network_run'")
 s=s.replace("R=Path('"+base+"')/sys.argv[1]/'network_run'","R=Path('"+base+"')/f'seed_{int(sys.argv[2])}'/sys.argv[1]/'network_run'")
 put(n.replace('480','481'),s)
put('verify_archive481.py',convert((T/'verify_archive480.py').read_text(encoding='utf-8'),480))
put('run_shadow481.sh','''#!/bin/bash
set -eu
cd /home/scv/shared/phase481_20261108
ARM="$1"
PY=/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python
trap 'code=$?; echo "$code" > "${ARM}_EXIT_CODE"' EXIT
for SEED in 45202 45203; do
 if [ "$ARM" = C ]; then
  $PY train_control481.py "$ARM" "$SEED"
 else
  $PY train_shadow481.py "$ARM" "$SEED"
 fi
 $PY verify_shadow481.py "$ARM" "$SEED"
 $PY boundary_shadow481.py "$ARM" "$SEED"
 if [ "$ARM" != C ]; then $PY ablate_shadow481.py "$ARM" "$SEED"; fi
 touch "seed_${SEED}/${ARM}/DONE"
done
''')
