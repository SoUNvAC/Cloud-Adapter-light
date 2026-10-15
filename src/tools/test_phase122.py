"""Synthetic CPU-only: precise threshold domain, strata and group uncertainty."""
import tempfile
from pathlib import Path
import numpy as np
from phase122_metrics import evaluate,operating,interval
from phase121_metrics import oracle
from test_phase121_metrics import main as exact_oracle_test
def main():
    exact_oracle_test()
    boundary=float(np.float32(.5));t=float(np.nextafter(np.float64(boundary),np.inf))
    op=operating(np.array([0,1,2]),np.array([.5,.5,.6],np.float32),t)
    assert op['FP']==0 and op['TP']==1 and op['Surface_FPR']==op['Cloud_FPR']==0
    assert operating(np.array([0,0]),np.array([.7,.8]),.5)['Cloud_FPR'] is None
    rng=np.random.default_rng(122015);g={}
    for i in range(5):
        y=np.array([0,1,2,0,1,2] if i else [0,1,0,1,0,1],np.uint8)
        g[str(i)]=dict(products=str(i),truth=y,R=rng.random(6).astype(np.float32),C=rng.random(6).astype(np.float32))
    base=Path(__file__).resolve().parents[2]/'outputs/phase122';base.mkdir(exist_ok=True,parents=True)
    with tempfile.TemporaryDirectory(dir=base) as tmp:
        assert Path(tmp).resolve().is_relative_to(base.resolve())
        s,c=evaluate(g,'test',{'R':t,'C':.7},tmp)
    assert len(s)==4 and len(c)==3 and s[0]['AP_groups']==4 and s[0]['negative_groups']==5
    assert s[0]['macro_AP_pp']==s[1]['macro_AP_pp'] and s[2]['macro_AP_pp']==s[3]['macro_AP_pp']
    assert all(v['AP']['mean']==0 for v in c[1:])
    assert interval([None,.1])['n']==1 and interval([None])['mean'] is None
    print(dict(status='passed',AP_threshold_invariance=True,float64_boundary_exclusion=True,negative_strata_NA=True,
        no_shadow_groups_in_FP=True,group_bootstrap=True,Omni_calls=0))
if __name__=='__main__':main()
