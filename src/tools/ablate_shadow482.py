"""Independent batch1 evaluation of best unconstrained and protected new networks."""
from pathlib import Path
import sys,os,json,hashlib,csv
import numpy as np
import torch
from PIL import Image
ARM=sys.argv[1];R=Path('/home/scv/shared/phase482_20261108')/ARM/'network_run';SRC=Path('/home/scv/Cloud-Adapter-light/src');P=Path('/home/scv/shared/phase431_20261102');sys.path[:0]=['/home/scv/shared/phase450_20261106',str(P/'code'),'/home/scv/shared/phase400_20261031/code'];os.chdir(SRC)
from infer_shadow482 import Frozen431WithOcclusion
def digest(m):
    h=hashlib.sha256()
    for n,t in m.state_dict().items():h.update(n.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def main():
    from mmengine.config import Config
    from mmseg.registry import MODELS
    from mmseg.utils import register_all_modules
    from report_phase400 import describe
    register_all_modules();import phase431.integration;torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    done=json.loads((R/'COMPLETE.json').read_text());assert done['passed'] and done['steps']==6000
    choices={'best_shadow':done['best_shadow']['step']}
    # Same trained checkpoint; only feature injection is switched off.
    baseline=json.loads(Path('/home/scv/shared/phase450_20261106/network_run/FINAL_RESULTS.json').read_text())['results'][0];rows=[r for r in json.loads(Path('/home/scv/shared/phase442_20261103/INPUTS.json').read_text()) if r['domain']=='A'];all_results=[]
    for step in sorted(set(choices.values())):
        cfg=Config.fromfile(str(P/'code/phase431/config.py'));cfg.model.backbone.variant='fused';cfg.model.backbone.repo='/home/scv/shared/phase429_20261102/dinov3';cfg.model.backbone.weights='/home/scv/shared/phase429_20261102/dinov3_splus_native.pth'
        teacher=MODELS.build(cfg.model);teacher.load_state_dict(torch.load(P/'fused/best_mIoU_iter_32000.pth',map_location='cpu')['state_dict'],strict=True)
        state=torch.load(R/f'branch_{step}.pth',map_location='cpu');loaded=teacher.load_state_dict(state['student_patch'],strict=False);assert not loaded.unexpected_keys and all(n.startswith('backbone.encoder.') for n in loaded.missing_keys);network=Frozen431WithOcclusion(teacher,state['state_dict']).cuda().eval();setattr(network.branch.redistribution,"enabled",False);before=digest(network)
        cm=np.zeros((3,3),np.int64);oldcm=np.zeros((3,3),np.int64);transition=np.zeros((3,3,3),np.int64);per=[];out=R/'ablation_off'/f'step_{step}';out.mkdir(parents=True,exist_ok=False)
        for row in rows:
            rgb=np.array(Image.open(row['image']).convert('RGB'));raw=np.array(Image.open(row['mask']));truth=raw.copy()
            for k,v in {0:0,1:1,2:1,3:2}.items():truth[raw==k]=v
            with torch.inference_mode():logits,base,delta,gate=network(torch.from_numpy(rgb).permute(2,0,1)[None].cuda());pred=logits.argmax(1)[0].cpu().numpy().astype(np.uint8);old=np.load(Path('/home/scv/shared/phase450_20261106/network_run/independent/step_6000')/f"{row['index']:03d}.npz")['pred']
            valid=truth!=255;y=truth[valid].astype(np.int64);p=pred[valid];q=old[valid];c=np.bincount(3*y+p,minlength=9).reshape(3,3);t=np.bincount(9*y+3*q+p,minlength=27).reshape(3,3,3)
            assert np.array_equal(c,np.array([[np.count_nonzero(valid&(truth==a)&(pred==b)) for b in range(3)] for a in range(3)]))
            assert np.array_equal(t.sum(1),c);cm+=c;transition+=t;oldcm+=t.sum(2)
            per.append(dict(index=row['index'],group=row['group'],confusion=c.tolist(),baseline_confusion=t.sum(2).tolist(),transition=t.tolist()))
            np.savez_compressed(out/f"{row['index']:03d}.npz",pred=pred)
        assert digest(network)==before;oldmetrics=describe(oldcm);metrics=describe(cm);assert max(abs(a-b) for a,b in zip(oldmetrics['IoU'],baseline['IoU']))<.002
        protected=all(a>=b for a,b in zip(metrics['IoU'][:2],baseline['IoU'][:2]));success=protected and metrics['IoU'][2]>=70
        damage=[int(transition[c,c].sum()-transition[c,c,c]) for c in range(3)];repaired=[int(transition[c,:,c].sum()-transition[c,c,c]) for c in range(3)]
        result=dict(step=step,roles=[k for k,v in choices.items() if v==step],**metrics,baseline=oldmetrics,protected=protected,success=success,IoU_delta=(np.array(metrics['IoU'])-np.array(oldmetrics['IoU'])).tolist(),original_correct_damaged=damage,original_wrong_repaired=repaired,transition=transition.tolist(),state_unchanged=True)
        groups=sorted({v['group'] for v in per});paired=np.array([[sum((np.array(v[k]) for v in per if v['group']==g),start=np.zeros((3,3),np.int64)) for k in ['baseline_confusion','confusion']] for g in groups]);rng=np.random.default_rng(44502);draws=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=2000);m=np.einsum('bn,nkij->bkij',draws,paired);diag=np.diagonal(m,axis1=-2,axis2=-1);iou=100*diag/(m.sum(-1)+m.sum(-2)-diag);diff=iou[:,1]-iou[:,0];result['paired_ROI_delta_CI95']=np.quantile(diff,[.025,.975],axis=0).T.tolist()
        np.savez_compressed(R/f'OFF_BOOTSTRAP_{step}.npz',groups=np.array(groups),confusions=paired,draws=draws,IoU_delta=diff)
        (R/f'OFF_INDEPENDENT_{step}.json').write_text(json.dumps(dict(result=result,per_image=per),indent=2));all_results.append(result)
        # Single combined state file for reproducible inference, kept on remoteD.
        torch.save(dict(state_dict=network.state_dict(),branch_step=step,architecture='Frozen431WithOcclusion',teacher_checkpoint=str(P/'fused/best_mIoU_iter_32000.pth')),out/'combined_network.pth')
        del network,teacher;torch.cuda.empty_cache()
    (R/'ABLATION_OFF.json').write_text(json.dumps(dict(passed=True,choices=choices,results=all_results,extra_forwards=535*len(all_results),target_label_supervision_percent=0),indent=2))
    print(json.dumps(all_results))
if __name__=='__main__':main()
