from pathlib import Path
import re
T=Path(__file__).parent
def put(n,s):(T/n).write_bytes(s.encode('utf-8'))
def phase(s,a,b):return re.sub(r'(?<![0-9a-fA-F])'+str(a)+r'(?![0-9a-fA-F])',str(b),s)
# Explicit phase/tool identifiers only: never rewrite numeric hashes or sample counts.
def swap(s,a,b):return s.replace('phase'+str(a),'phase'+str(b)).replace('shadow'+str(a),'shadow'+str(b)).replace('redistribution'+str(a),'redistribution'+str(b)).replace('region'+str(a),'region'+str(b)).replace('effect'+str(a),'effect'+str(b)).replace('archive'+str(a),'archive'+str(b)).replace('all'+str(a),'all'+str(b))
base='/home/scv/shared/phase494_20261109'
for stem in ['train_shadow','infer_shadow','verify_shadow','ablate_shadow','ablate_region','boundary_shadow','boundary_effect','boundary_region','verify_archive']:
 s=swap((T/f'{stem}493.py').read_text(encoding='utf-8'),493,494)
 if stem=='train_shadow':
  s=s.replace("ARM=sys.argv[1];assert ARM in ['A','B']","ARM=sys.argv[1];assert ARM=='B'\nSEED=int(sys.argv[2]);assert SEED in [45202,45203]\nINIT_SEED=48000+SEED-45200")
  s=s.replace("ROOT=Path('"+base+"')","ROOT=Path('"+base+"')/f'seed_{SEED}'")
  s=s.replace('45201','SEED').replace('torch.manual_seed(48001)','torch.manual_seed(INIT_SEED)').replace('init_seed=48001','init_seed=INIT_SEED')
 else:
  s=s.replace("ARM=sys.argv[1];R=Path('"+base+"')/ARM/'network_run'","ARM=sys.argv[1];SEED=int(sys.argv[2]);R=Path('"+base+"')/f'seed_{SEED}'/ARM/'network_run'")
  s=s.replace("R=Path('"+base+"')/sys.argv[1]/'network_run'","R=Path('"+base+"')/f'seed_{int(sys.argv[2])}'/sys.argv[1]/'network_run'")
 put(f'{stem}494.py',s)
put('shadow_redistribution494.py',(T/'shadow_redistribution493.py').read_text(encoding='utf-8'))
for stem in ['preflight_shadow','summarize_shadow','finish_shadow','status_shadow','report_shadow']:
 s=swap((T/f'{stem}489.py').read_text(encoding='utf-8'),489,494).replace('phase494_20261108','phase494_20261109')
 if stem=='preflight_shadow':
  s=s.replace('phase488_20261108/shadow_redistribution488.py','phase493_20261109/shadow_redistribution493.py')
  s=s.replace("'phase488_20261108' if method=='N' else 'phase482_20261108'","'phase493_20261109' if method=='N' else 'phase488_20261108'").replace('phase483_20261108','phase489_20261108')
 if stem=='status_shadow':s=s.replace('phase483_20261108','phase489_20261108')
 if stem=='summarize_shadow':s=s.replace('482B fine redistribution baseline','488B raw-gradient texture baseline').replace('488B raw-gradient texture evidence','493B pre-pooling amplitude evidence')
 if stem=='report_shadow':
  s=s.replace('Phase489','Phase494').replace('*489.py','*494.py').replace('*489.sh','*494.sh')
  s=s.replace("if n!='branch_0.pth':assert sha(R/f'seed_{seed}'/'B/network_run'/n)==w['sha256']","assert len(w['sha256'])==64 and v['passed']")
  s=s.replace('482B/483基线','488B/489基线').replace('488B/489纹理分支','493B/494细纹理分支')
  s=s.replace('Phase488 B三seed复现','Phase493 B三seed复现')
  start=s.index(" '网络与Phase488逐字节一致，");end=s.index("if (R/'INTERPRETATION.md')",start)
  s=s[:start]+" '网络模块与Phase493逐字节一致，固定B亮度差幅度编码：原图1/2/4像素偏移比较，tanh(20×差值)后再池化。复用45201/48001，新增45202/48002、45203/48003；与488B/489同seed基线配对。固定431＋450起点，6000步、batch4，骨干/450/BN冻结，CloudSEN训练8490/原始验证535、107ROI，目标监督0%，固定argmax。','']\n"+s[end:]
  s=s.replace('17个新增模块参数张量','结构模块21个参数张量（保留17个、新增4个）').replace('selected_weights_verified=True,','remote_weights_verified=True,local_weights_complete=False,')
 put(f'{stem}494.py',s)
s=swap((T/'collect_all493.py').read_text(),493,494);put('collect_all494.py',s)
s=swap((T/'collect_metadata493.py').read_text(),493,494);put('collect_metadata494.py',s)
put('run_shadow494.sh','''#!/bin/bash
set -eu
cd /home/scv/shared/phase494_20261109
SEED="$1"
PY=/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python
trap 'code=$?; echo "$code" > "seed_${SEED}_EXIT_CODE"' EXIT
$PY train_shadow494.py B "$SEED"
$PY verify_shadow494.py B "$SEED"
$PY boundary_shadow494.py B "$SEED"
$PY ablate_shadow494.py B "$SEED"
$PY boundary_effect494.py B "$SEED"
$PY ablate_region494.py B "$SEED"
$PY boundary_region494.py B "$SEED"
touch "seed_${SEED}/B/DONE"
''')
