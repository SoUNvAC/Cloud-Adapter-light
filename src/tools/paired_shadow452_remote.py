from pathlib import Path
import json
import numpy as np
R=Path('/home/scv/shared/phase452_20261106')
def main():
 root450=Path('/home/scv/shared/phase450_20261106/network_run');models={'450':np.load(root450/'BOOTSTRAP_6000.npz')};steps={'450':6000}
 for arm in ['A','B']:
  d=R/arm/'network_run';f=json.loads((d/'FINAL_RESULTS.json').read_text())
  for role,step in f['choices'].items():models[arm+'_'+role]=np.load(d/f'BOOTSTRAP_{step}.npz');steps[arm+'_'+role]=step
 output={}
 comparisons=[(k,'450') for k in models if k!='450']+[('B_best_shadow','A_best_shadow'),('B_best_protected','A_best_protected')]
 for new,old in comparisons:
  a=models[old];b=models[new];assert np.array_equal(a['groups'],b['groups']);pair=np.stack([a['confusions'][:,1],b['confusions'][:,1]],axis=1)
  m=np.einsum('bn,nkij->bkij',b['draws'],pair);d=np.diagonal(m,axis1=-2,axis2=-1);iou=100*d/(m.sum(-1)+m.sum(-2)-d);diff=iou[:,1]-iou[:,0]
  output[new+'_minus_'+old]=dict(steps=[steps[new],steps[old]],IoU_delta_CI95=np.quantile(diff,[.025,.975],axis=0).T.tolist(),mIoU_delta_CI95=np.quantile(diff.mean(1),[.025,.975]).tolist())
 (R/'PAIRED_COMPARISON.json').write_text(json.dumps(output,indent=2));print(json.dumps(output))
if __name__=='__main__':main()
