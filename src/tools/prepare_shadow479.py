from pathlib import Path
T=Path(__file__).parent
def convert(s):return s.replace('476','479').replace('phase479_20261107','phase479_20261108').replace('material','envelope').replace('MaterialConvolution','ShadowEnvelope').replace('chroma_mode','morphology_mode').replace('chroma=','morphology=')
def put(n,s):(T/n).write_bytes(s.encode('utf-8'))
s=convert((T/'train_shadow476.py').read_text(encoding='utf-8'))
s=s.replace("branch_updated=ARM=='B'","branch_updated=True")
s=s.replace('zero_difference_kernel=True','zero_output_projection=True')
s=s.replace('New spatial envelope convolution','New multiscale dark-envelope branch')
s=s.replace("chroma=ARM=='B'","morphology=ARM=='B'")
put('train_shadow479.py',s)
for n in ['infer_shadow476.py','ablate_shadow476.py','collect_shadow476.py','boundary_shadow476.py']:
 put(n.replace('476','479'),convert((T/n).read_text(encoding='utf-8')))
s=(T/'verify_shadow478.py').read_text(encoding='utf-8').replace('478','479');put('verify_shadow479.py',s)
put('verify_archive479.py',(T/'verify_archive478.py').read_text(encoding='utf-8').replace('478','479'))
put('run_shadow479.sh','''#!/bin/bash
set -eu
cd /home/scv/shared/phase479_20261108
PY=/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python
$PY train_shadow479.py "$1"
$PY verify_shadow479.py "$1"
$PY boundary_shadow479.py "$1"
$PY ablate_shadow479.py "$1"
''')
