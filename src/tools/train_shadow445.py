"""Frozen Phase431, source-only6000-step local occlusion branch experiment."""
from pathlib import Path
import sys,os,json,hashlib,time,traceback,random,math
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from shadow_branch445 import LocalOcclusionBranch,ring
R=Path('/home/scv/shared/phase445_20261106/network_run');P=Path('/home/scv/shared/phase431_20261102');SRC=Path('/home/scv/Cloud-Adapter-light/src');sys.path[:0]=[str(P/'code'),'/home/scv/shared/phase400_20261031/code'];os.chdir(SRC)
def save(n,x):
    p=R/n;p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False));q.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(model):
    h=hashlib.sha256()
    for n,t in model.state_dict().items():h.update(n.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
class Images(Dataset):
    def __init__(self,split):
        self.root=SRC/'data/cloudsen12_high_l1c';self.files=sorted((self.root/'img_dir'/split).glob('*.png'));self.split=split
    def __len__(self):return len(self.files)
    def __getitem__(self,i):
        path=self.files[i];rgb=np.array(Image.open(path).convert('RGB'));raw=np.array(Image.open(self.root/'ann_dir'/self.split/path.name));y=raw.copy()
        for k,v in {0:0,1:1,2:1,3:2}.items():y[raw==k]=v
        assert rgb.shape==(512,512,3)
        if self.split=='train':
            if random.random()<.5:rgb=rgb[:,::-1];y=y[:,::-1]
            if random.random()<.5:rgb=rgb[::-1];y=y[::-1]
        return torch.from_numpy(rgb.copy()).permute(2,0,1),torch.from_numpy(y.copy()).long(),path.name
def main():
    from mmengine.config import Config
    from mmseg.registry import MODELS
    from mmseg.utils import register_all_modules
    from report_phase400 import describe
    R.mkdir(exist_ok=False);torch.set_num_threads(4);torch.manual_seed(44501);np.random.seed(44501);random.seed(44501);torch.backends.cudnn.benchmark=False
    register_all_modules();import phase431.integration
    cfg=Config.fromfile(str(P/'code/phase431/config.py'));cfg.model.backbone.variant='fused';cfg.model.backbone.repo='/home/scv/shared/phase429_20261102/dinov3';cfg.model.backbone.weights='/home/scv/shared/phase429_20261102/dinov3_splus_native.pth'
    checkpoint=P/'fused/best_mIoU_iter_32000.pth';teacher=MODELS.build(cfg.model);teacher.load_state_dict(torch.load(checkpoint,map_location='cpu')['state_dict'],strict=True);teacher.cuda().eval().requires_grad_(False);teacher_hash=digest(teacher)
    captured={}
    teacher.backbone.spatial.register_forward_hook(lambda m,a,o:captured.update(detail=o[1]))
    teacher.backbone.register_forward_hook(lambda m,a,o:captured.update(features=o))
    branch=LocalOcclusionBranch().cuda();optimizer=torch.optim.AdamW(branch.parameters(),lr=2e-4,weight_decay=.01)
    assert {id(p) for g in optimizer.param_groups for p in g['params']}=={id(p) for p in branch.parameters()}
    mean=torch.tensor([123.675,116.28,103.53],device='cuda')[None,:,None,None];std=torch.tensor([58.395,57.12,57.375],device='cuda')[None,:,None,None]
    train=Images('train');val=Images('val');assert len(train)==8490 and len(val)==535
    loader=DataLoader(train,batch_size=4,shuffle=True,num_workers=4,pin_memory=True,persistent_workers=True,drop_last=False);vl=DataLoader(val,batch_size=4,shuffle=False,num_workers=2,pin_memory=True,persistent_workers=True)
    assert not set(p.resolve() for p in train.files)&set(p.resolve() for p in val.files)
    for outer,inner in [(7,3),(15,7),(31,15)]:assert torch.allclose(ring(torch.ones(1,1,40,40,device='cuda'),outer,inner),torch.ones(1,1,40,40,device='cuda'),atol=1e-5)
    baseline=json.loads(Path('/home/scv/shared/phase431_20261102/cloudsen_original_recheck_20261106/RESULT.json').read_text());floor=np.array(baseline['IoU'][:2]);calls=0
    def forward(rgb):
        nonlocal calls
        rgb=rgb.cuda(non_blocking=True).float();b=len(rgb)
        with torch.no_grad():scores=teacher.encode_decode((rgb-mean)/std,[dict(img_shape=(512,512),ori_shape=(512,512),pad_shape=(512,512),padding_size=[0,0,0,0]) for _ in range(b)]).clamp_min(1e-8);base=(scores/scores.sum(1,keepdim=True)).log()
        calls+=b;return (*branch(captured['detail'],captured['features'],rgb/255,base),base)
    history=[];best_shadow=-1.;best_protected=-1.;start=time.monotonic()
    def evaluate(step):
        nonlocal best_shadow,best_protected
        branch.eval();cm=np.zeros((3,3),np.int64);teacher_cm=np.zeros((3,3),np.int64);transition=np.zeros((3,3,3),np.int64);per=[]
        with torch.no_grad():
            for rgb,y,names in vl:
                logits,delta,gate,base=forward(rgb);pred=logits.argmax(1).cpu().numpy();old=base.argmax(1).cpu().numpy();truth=y.numpy()
                if step==0:assert np.array_equal(pred,old) and delta.count_nonzero()==0
                for t,p,q,n in zip(truth,pred,old,names):
                    valid=t!=255;t=t[valid].astype(np.int64);p=p[valid];q=q[valid];c=np.bincount(t*3+p,minlength=9).reshape(3,3);cm+=c;teacher_cm+=np.bincount(t*3+q,minlength=9).reshape(3,3);transition+=np.bincount(t*9+q*3+p,minlength=27).reshape(3,3,3);per.append(dict(name=n,confusion=c.tolist()))
        assert np.max(np.abs(np.array(describe(teacher_cm)['IoU'])-baseline['IoU']))<.002
        metrics=describe(cm);protected=bool(np.all(np.array(metrics['IoU'][:2])>=floor));success=protected and metrics['IoU'][2]>=70
        row=dict(step=step,**metrics,protected=protected,success=success,teacher=describe(teacher_cm),transition=transition.tolist(),seconds=time.monotonic()-start)
        if step>0:
            torch.save(dict(state_dict=branch.state_dict(),step=step,teacher_sha=sha(checkpoint)),R/f'branch_{step}.pth')
            if metrics['IoU'][2]>best_shadow:best_shadow=metrics['IoU'][2];save('BEST_SHADOW.json',row)
            if protected and metrics['IoU'][2]>best_protected:best_protected=metrics['IoU'][2];save('BEST_PROTECTED.json',row)
        history.append(row);save(f'validation/step_{step}.json',dict(metrics=row,per_image=per));save('HISTORY.json',history);branch.train();return row
    save('LOCK.json',dict(teacher=str(checkpoint),teacher_sha=sha(checkpoint),teacher_state_sha=teacher_hash,branch_parameters=sum(p.numel() for p in branch.parameters()),steps=6000,batch_size=4,seed=44501,lr=2e-4,train_images=8490,val_images=535,target_label_supervision_percent=0,BN='Entire original model fixed eval; new branch uses no BN',augmentation='Source-only random horizontal and vertical flips; no color alteration',loss='Three-class weightedCE +2*correct-nonshadow KL +2*softIoU protection hinge +0.01*delta squared',selection='Save best shadow and separately best with both nonshadow IoUs >= fixed baseline',code_sha={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('shadow_branch445.py')]}))
    zero=evaluate(0);save('ZERO_INIT_CHECK.json',dict(passed=True,images=535,exact_teacher_argmax=True,zero_delta=True,metrics=zero))
    initial={n:p.detach().cpu().clone() for n,p in branch.named_parameters()};iterator=iter(loader)
    for step in range(1,6001):
        try:rgb,y,_=next(iterator)
        except StopIteration:iterator=iter(loader);rgb,y,_=next(iterator)
        y=y.cuda(non_blocking=True);logits,delta,gate,base=forward(rgb);valid=y!=255;old=base.argmax(1);hard=(old!=y)&((old==2)|(y==2))&valid
        weights=torch.ones_like(y,dtype=torch.float);weights[hard]=4.;weights[(y==2)&(old==2)]=1.5
        ce=F.cross_entropy(logits,y,ignore_index=255,reduction='none');loss_ce=(ce*weights*valid).sum()/(weights*valid).sum()
        prob=logits.softmax(1);oldprob=base.softmax(1);keep=valid&(old==y)&(y!=2);kl=(oldprob*(base-logits.log_softmax(1))).sum(1);loss_keep=(kl*keep).sum()/keep.sum().clamp_min(1)
        softloss=0.
        for c in [0,1]:
            gt=(y==c)&valid;p=prob[:,c]*valid;q=oldprob[:,c]*valid
            a=(p*gt).sum()/((p+gt-p*gt).sum()+1);b=(q*gt).sum()/((q+gt-q*gt).sum()+1);softloss=softloss+F.relu(b-a)
        loss=loss_ce+2*loss_keep+2*softloss+.01*delta.square().mean();assert torch.isfinite(loss)
        lr=2e-4*min(step/200,1)*(1-step/6001)**.9
        for g in optimizer.param_groups:g['lr']=lr
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(branch.parameters(),1.);optimizer.step()
        if step in [10,100] or step%1000==0:
            assert digest(teacher)==teacher_hash and all(p.grad is None for p in teacher.parameters()) and all(not m.training for m in teacher.modules())
            changed=[n for n,p in branch.named_parameters() if not torch.equal(p.detach().cpu(),initial[n])];assert changed and all(torch.isfinite(p).all() for p in branch.parameters());save('UPDATE_CHECK.json',dict(step=step,teacher_unchanged=True,changed=changed))
        if step%50==0:save('STATUS.json',dict(step=step,total=6000,loss=float(loss.detach()),CE=float(loss_ce.detach()),keep_KL=float(loss_keep.detach()),mean_abs_delta=float(delta.detach().abs().mean()),gate_mean=float(gate.detach().mean()),seconds=time.monotonic()-start,forward_images=calls))
        if step%1000==0:evaluate(step)
    assert digest(teacher)==teacher_hash
    save('COMPLETE.json',dict(passed=True,steps=6000,teacher_unchanged=True,seconds=time.monotonic()-start,forward_images=calls,best_shadow=json.loads((R/'BEST_SHADOW.json').read_text()),best_protected=json.loads((R/'BEST_PROTECTED.json').read_text()) if (R/'BEST_PROTECTED.json').exists() else None))
if __name__=='__main__':
    try:main()
    except BaseException:
        R.mkdir(exist_ok=True);save('FAILURE.json',dict(error=traceback.format_exc()));raise
