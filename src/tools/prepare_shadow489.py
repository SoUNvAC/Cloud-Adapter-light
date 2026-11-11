from pathlib import Path
T=Path(__file__).parent
def put(n,s):(T/n).write_bytes(s.encode('utf-8'))
base='/home/scv/shared/phase489_20261108'
for n in ['train_shadow488.py','infer_shadow488.py','verify_shadow488.py','ablate_shadow488.py','ablate_region488.py','boundary_shadow488.py','boundary_effect488.py','boundary_region488.py','verify_archive488.py']:
 s=(T/n).read_text(encoding='utf-8').replace('488','489')
 if n=='train_shadow488.py':
  s=s.replace("ARM=sys.argv[1];assert ARM in ['A','B']", "ARM=sys.argv[1];assert ARM=='B'\nSEED=int(sys.argv[2]);assert SEED in [45202,45203]\nINIT_SEED=48000+SEED-45200")
  s=s.replace("ROOT=Path('"+base+"')", "ROOT=Path('"+base+"')/f'seed_{SEED}'")
  s=s.replace('45201','SEED').replace('torch.manual_seed(48001)','torch.manual_seed(INIT_SEED)').replace('init_seed=48001','init_seed=INIT_SEED')
 else:
  s=s.replace("ARM=sys.argv[1];R=Path('"+base+"')/ARM/'network_run'", "ARM=sys.argv[1];SEED=int(sys.argv[2]);R=Path('"+base+"')/f'seed_{SEED}'/ARM/'network_run'")
  s=s.replace("R=Path('"+base+"')/sys.argv[1]/'network_run'", "R=Path('"+base+"')/f'seed_{int(sys.argv[2])}'/sys.argv[1]/'network_run'")
 put(n.replace('488','489'),s)
put('shadow_redistribution489.py',(T/'shadow_redistribution488.py').read_text(encoding='utf-8'))
s=(T/'accept_shadow483.py').read_text(encoding='utf-8').replace('483','489')
s=s.replace("assert set(origin['state_dict'])==set(control['state_dict'])|{'redistribution.contrast_mode'}", "assert len(set(origin['state_dict'])-set(control['state_dict']))==5\n assert not bool(origin['state_dict']['redistribution.normalized_mode'])\n assert lock['new_module']['normalized']==False and lock['new_module']['texture_scales_pixels']==[12,28]")
s=s.replace('len(new)==13','len(new)==17')
put('accept_shadow489.py',s)
for n in ['finish_shadow483.py','collect_all483.py','status_shadow483.py']:
 s=(T/n).read_text(encoding='utf-8').replace('483','489')
 if n=='collect_all483.py':s=s.replace("{'independent','ablation_off'}","{'independent','ablation_off','region_off'}")
 if n=='status_shadow483.py':s=s.replace('phase481_20261108','phase483_20261108')
 put(n.replace('483','489'),s)
s=(T/'summarize_shadow483.py').read_text(encoding='utf-8').replace('483','489')
s=s.replace("['C','S','N']","['S','N']").replace("[('N','S'),('N','C'),('S','C')]","[('N','S')]")
s=s.replace("methods={'C':'Decoder only','S':'Redistribution with 0.1 coverage auxiliary BCE','N':'Redistribution without auxiliary BCE'}","methods={'S':'482B fine redistribution baseline','N':'488B raw-gradient texture evidence'}")
s=s.replace("onoff=[]","onoff=[];texture_onoff=[]")
needle="   onoff.append(dict(seed=seed,step=step,IoU_delta=(np.array(on['IoU'])-off['IoU']).tolist(),mIoU_delta=on['mIoU']-off['mIoU']))"
s=s.replace(needle,needle+"\n   texture_off=validate(d/f'REGION_OFF_INDEPENDENT_{step}.json')\n   texture_onoff.append(dict(seed=seed,step=step,IoU_delta=(np.array(on['IoU'])-texture_off['IoU']).tolist(),mIoU_delta=on['mIoU']-texture_off['mIoU']))")
s=s.replace('on_off_best_shadow=onoff,','on_off_best_shadow=onoff,texture_on_off_best_shadow=texture_onoff,')
put('summarize_shadow489.py',s)
put('run_shadow489.sh','''#!/bin/bash
set -eu
cd /home/scv/shared/phase489_20261108
SEED="$1"
PY=/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python
trap 'code=$?; echo "$code" > "seed_${SEED}_EXIT_CODE"' EXIT
$PY train_shadow489.py B "$SEED"
$PY verify_shadow489.py B "$SEED"
$PY boundary_shadow489.py B "$SEED"
$PY ablate_shadow489.py B "$SEED"
$PY boundary_effect489.py B "$SEED"
$PY ablate_region489.py B "$SEED"
$PY boundary_region489.py B "$SEED"
touch "seed_${SEED}/B/DONE"
''')
