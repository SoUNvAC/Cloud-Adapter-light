"""CPU-only gates: exact accepted operators plus within-group seed averaging."""
import tempfile,json,csv
from pathlib import Path
import numpy as np
from phase123_metrics import evaluate,engine
from test_phase122 import main as accepted_tests
def main():
    accepted_tests();groups={}
    for i in range(4):
        y=np.array([0,1,2,2] if i else [0,1,0,1],np.uint8)
        groups[str(i)]=dict(products=str(i),truth=y,R=np.array([.1,.6,.4,.7],np.float32),
            C122=np.array([.1,.3,.8,.6],np.float32),C123=np.array([.2,.5,.6,.9],np.float32))
    out=Path(__file__).resolve().parents[2]/'outputs/phase123';out.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as d:
        s,m=evaluate(groups,'Catalogue',dict(R=.6,C122=.6,C123=.7),d)
        assert len(s)==6 and m['recall']['n']==3 and m['FPR']['n']==4
        with (Path(d)/'Catalogue_per_group.csv').open(encoding='utf-8',newline='') as f:rr=list(csv.DictReader(f))
        state={(v['group_id'],v['setting']):v for v in rr};expected=[]
        for gid in ('1','2','3'):
            a,b,r=[float(state[gid,v]['recall']) for v in ('C122_cal','C123_cal','R_cal')];expected.append(((a+b)/2-r)*100)
        assert abs(m['recall']['mean']-sum(expected)/3)<1e-12
        assert m['recall']['CI95']==engine.interval(expected)['CI95']
        assert len({v['group_id'] for v in rr})==4
    print('Phase123 exact-boundary, synchronized group bootstrap and seed-metric-mean CPU tests passed; 0 model calls')
if __name__=='__main__':main()
