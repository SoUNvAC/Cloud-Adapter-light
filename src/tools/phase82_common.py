"""Phase82 paths, bindings and fixed descriptive group bootstrap."""
from phase80_common import *
ROOT=DATA/'phase82_20261011'
WORK=REPO/'src/work_dirs/phase82_20261011'
PREVIOUS=DATA/'phase80_20261011'
PHASE81=DATA/'phase81_20261011'
RUNCONTROL=Path('/home/scv/shared/phase82_control_20261011')
PROTOCOL=REPO/'src/research_plans/PHASE82_NONSHADOW_DISTILLATION_20261011.md'

def interval(values):
    x=np.asarray(values,np.float64)
    assert x.ndim==1 and np.isfinite(x).all()
    result=dict(mean=float(x.mean()) if len(x) else None,ci95=None,upper95_one_sided=None,
                n_groups=len(x),seed=82011,resamples=10000,descriptive_only=True)
    if len(x)>=2:
        bs=x[np.random.default_rng(82011).integers(0,len(x),(10000,len(x)))].mean(1)
        result.update(ci95=np.quantile(bs,[.025,.975]).tolist(),upper95_one_sided=float(np.quantile(bs,.95)))
    return result

def continuation_gate(delta_miou_pooled,delta_AP,delta_recall,delta_miou_macro,fpr):
    checks=dict(pooled_mIoU_gain_at_least_1pp=delta_miou_pooled>=1,
                macro_AP_loss_at_most_1pp=delta_AP>=-1,
                macro_recall_loss_at_most_1pp=delta_recall>=-1,
                macro_mIoU_loss_at_most_1pp=delta_miou_macro>=-1,
                mean_FPR_at_most_1percent=fpr<=.01)
    return dict(checks=checks,retain_fixed_scheme=all(checks.values()),formal_confirmation=False)
