"""Phase80 scope, exact state conversion checks and descriptive statistics."""
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
from prepare_phase65a_explore import digest, SOURCE_SHA, SPLIT_SHA

REPO=Path('/home/scv/Cloud-Adapter-light')
DATA=Path('/home/scv/shared/data/sentinel2_cloud_mask_catalogue_4172871')
ROOT=DATA/'phase80_20261011'
WORK=REPO/'src/work_dirs/phase80_20261011'
PREP=DATA/'phase65a_explore_20261008'
SOURCE=REPO/'src/work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth'
RECOVERY=DATA/'phase65d_recovery_20261008'
OLD=DATA/'phase65d_20261008_recovery01'
CONTROL=DATA/'phase67_20261010'
PROTOCOL=REPO/'src/research_plans/PHASE80_ADAPTATION_BRANCH_20261011.md'
MANIFEST_SHA='afe39e4af254d7dad356cacdf594bf19d3d501a829e67d055811da4db0a9a3eb'
RECOVERY_SHA='bcb91cf56d3b796c22bbf60891fb8d045dd8e6af80f5c42a00818fa72501f31f'

def read(p): return json.loads(Path(p).read_text())
def write(p,x):
    with Path(p).open('x',encoding='utf-8') as f: json.dump(x,f,indent=2,allow_nan=False)

def deadline():
    if datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat()>='2026-10-20':
        raise RuntimeError('Research training deadline reached')

def permitted_lora(name):
    parts=name.split('.')
    return (len(parts)==6 and parts[:2]==['backbone','blocks'] and parts[2].isdigit()
            and ((parts[3]=='attn' and parts[4] in ('qkv','proj')) or
                 (parts[3]=='mlp' and parts[4] in ('fc1','fc2')))
            and parts[5] in ('lora_a','lora_b'))

def state_keys(source,target):
    extra=set(target)-set(source); absent=set(source)-set(target)
    if absent or not extra or any(not permitted_lora(k) for k in extra):
        raise ValueError('Source/LoRA key mismatch: '+str((sorted(absent),sorted(extra))))
    for k in source:
        if tuple(source[k].shape)!=tuple(target[k].shape):raise ValueError('Shape drift '+k)
    return sorted(extra)

def base_check():
    assert digest(SOURCE)==SOURCE_SHA
    assert digest(REPO/'src/work_dirs/phase65a/split_lock.json')==SPLIT_SHA
    assert digest(PREP/'manifest.json')==MANIFEST_SHA
    assert digest(RECOVERY/'recovery_lock.json')==RECOVERY_SHA
    assert digest(CONTROL/'control_lock.json')=='c2a83964073918015460e0ad89a9857b60b6a764e0f8562652472b224915f5ad'
    assert digest(CONTROL/'delivery/phase67_report.json')=='49e50f4f1bea2ef71e5dc533846243dd0fbc543e32adb0b46cc6529185afab71'
    assert digest(RECOVERY/'recovery_report.json')=='5e48cd76418d45a3bf6ee1c056f63792aa893e03aef43885cd37171c67030e8c'
    md=read(PREP/'manifest.json')
    assert md['scope']=='exploratory_fit_development_only'
    fit=[r for r in md['rows'] if r['split']=='fit']
    dev=[r for r in md['rows'] if r['split']=='development_val']
    assert len(fit)==155 and len({r['group_id'] for r in fit})==153
    assert len(dev)==len({r['group_id'] for r in dev})==32
    assert not {r['group_id'] for r in fit}&{r['group_id'] for r in dev}
    assert sum(r['is_representative'] for r in fit)==153
    return md

def interval(x):
    x=np.asarray(x,np.float64)
    assert x.ndim==1 and np.isfinite(x).all()
    out=dict(n_groups=len(x),mean=float(x.mean()) if len(x) else None,
             ci95=None,upper95_one_sided=None,resamples=10000,seed=80011,
             descriptive_only=True,conditional_on_single_training_and_development_selection=True)
    if len(x)>=2:
        rng=np.random.default_rng(80011)
        bs=x[rng.integers(0,len(x),(10000,len(x)))].mean(1)
        out.update(ci95=np.quantile(bs,[.025,.975]).tolist(),upper95_one_sided=float(np.quantile(bs,.95)))
    return out
