"""Phase85 fixed seed replication, inherited Phase80 settings."""
from phase80_common import *
import os
P80=ROOT
PHASE81=DATA/'phase81_20261011'
PARENT=DATA/'phase85_20261011_run02'
CONTROL85=Path('/home/scv/shared/phase85_control_20261011_run02')
SEED=int(os.environ.get('PHASE85_SEED','66'))
assert SEED in (66,67)
ROOT=PARENT/('seed'+str(SEED))
WORK=REPO/'src/work_dirs/phase85_20261011_run02'/('seed'+str(SEED))
PROTOCOL=REPO/'src/research_plans/PHASE85_LORA_SEED_REPLICATION_20261011.md'

def interval(values):
    x=np.asarray(values,np.float64)
    assert len(x)>0 and np.isfinite(x).all()
    means=x[np.random.default_rng(85011).integers(0,len(x),(10000,len(x)))].mean(1)
    return dict(mean=float(x.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),
        upper95_one_sided=float(np.quantile(means,.95)),n_groups=len(x),seed=85011,resamples=10000,descriptive_only=True)

def gate(policies):
    mm=policies['MM'];ls=[policies['LL66'],policies['LL67']];lm=[policies['LM66'],policies['LM67']]
    gains=[100*(p['macro_ap']['mean']-mm['macro_ap']['mean']) for p in ls]
    rules=dict(both_AP_gains_positive=all(v>0 for v in gains),mean_new_seed_AP_gain_at_least_1pp=float(np.mean(gains))>=1,
      each_recall_loss_at_most_1pp=all(100*(p['macro_recall']['mean']-mm['macro_recall']['mean'])>=-1 for p in ls),
      each_macro_FPR_at_most_1percent=all(p['macro_fpr']['mean']<=.01 for p in ls),
      each_LM_pooled_mIoU_at_least_MM=all(p['miou_percent']>=mm['miou_percent'] for p in lm),
      each_LM_macro_mIoU_loss_at_most_1pp=all(p['macro_miou_percent']['mean']-mm['macro_miou_percent']['mean']>=-1 for p in lm))
    return rules
