"""No new inference/fitting: verify per-pixel AP/confusions and fit operating points."""
import io,hashlib,zipfile,sys
import numpy as np
from phase80_common import *
from run_phase65d_recovery import records,prepare_verify,load,features
from phase65d_recovery_math import logit_score
from verify_phase78_local import independent_AP

def calc(x,m):
    return (((np.asarray(x)-np.asarray(m['mean']))/np.asarray(m['scale']))@np.asarray(m['theta'][:-1])+m['theta'][-1]).astype(np.float32)

def index(root,rel):return {r['product']:r for r in read(root/rel)['rows']}

def capacity():
    from run_phase80_training import setup
    setup()
    from mmengine.config import Config
    from mmseg.registry import MODELS
    cfg=Config.fromfile(str(REPO/'src/configs/protocol/phase65a_explore_rgb.py'))
    cfg.model.backbone.init_cfg=None
    model=MODELS.build(cfg.model);model.train()
    names={n:p.numel() for n,p in model.named_parameters() if p.requires_grad}
    assert names and all(n.startswith(('backbone.target_msre.','backbone.target_head_delta.')) for n in names)
    write(ROOT/'capacity_audit.json',dict(msre_trainable_parameters=sum(names.values()),msre_trainable_tensors=names,
        lora_trainable_parameters=read(ROOT/'preflight.json')['trainable_parameters'],
        checkpoint_hash=read(RECOVERY/'recovery_lock.json')['msre_checkpoint_sha256'],
        capacity_matched=False,note='Architecture parameter audit only; no Source/MsRE forward, fitting or checkpoint change'))

def main():
    base_check();rep=read(ROOT/'phase80_report.json');cl=read(ROOT/'calibration_lock.json')
    base=read(RECOVERY/'recovery_lock.json');ctl=read(CONTROL/'control_lock.json')
    li=index(ROOT,'development_val/index.json');ss=index(OLD,'development_val/index_source.json');ms=index(OLD,'development_val/index_msre.json')
    refs={n:{r['product']:r for r in v['rows']} for n,v in rep['policies'].items()}
    checks=[]
    with zipfile.ZipFile(DATA/'subscenes.zip') as z:
        for product,r in records('development_val').items():
            prepare_verify(r);truth=np.load(PREP/r['mask_path']);valid=truth!=255;y=truth[valid]==2
            b=z.read('subscenes/'+product+'.npy');assert hashlib.sha256(b).hexdigest()==r['member_sha256']['image']
            raw=np.load(io.BytesIO(b),allow_pickle=False)
            s=load(OLD,ss[product],'score');m=load(OLD,ms[product],'score');l=load(ROOT,li[product],'score')
            x=features(raw,m,s,valid);lx=np.column_stack([logit_score(l[valid]),x[:,2:5]]).astype(np.float32)
            ns=load(OLD,ms[product],'nonshadow')[valid];lns=load(ROOT,li[product],'nonshadow')[valid]
            scores={'Source':s[valid],'MsRE':m[valid],'LoRA_argmax':l[valid]}
            preds={'Source':load(OLD,ss[product],'pred')[valid],'MsRE':load(OLD,ms[product],'pred')[valid],
                   'LoRA_argmax':load(ROOT,li[product],'pred')[valid]}
            for n,xx,model,other in [('M',x[:,[0]],base['models']['msre_calibration'],ns),
                    ('M+S',x[:,[0,2,3,4]],ctl['models']['MsRE_spectral'],ns),
                    ('L',lx[:,[0]],cl['models']['L'],lns),('L+S',lx,cl['models']['L+S'],lns),
                    ('L3',x,base['models']['msre_source_spectral'],ns)]:
                scores[n]=calc(xx,model);preds[n]=np.where(scores[n]>=model['threshold'],2,other).astype(np.uint8)
            for n in rep['policies']:
                cm=np.array([[int(np.sum((truth[valid]==a)&(preds[n]==b))) for b in range(3)] for a in range(3)])
                assert np.array_equal(cm,np.asarray(refs[n][product]['confusion'])),(n,product)
                ap=independent_AP(y,scores[n]);ref=refs[n][product]['ranking']['ap']
                assert (ap is None and ref is None) or abs(ap-ref)<1e-12,(n,product,ap,ref)
                checks.append(dict(policy=n,product=product,AP=ap,confusion_identical=True))
            print('Phase80 independent shared pixel check',product,flush=True)
    audit=read(ROOT/'fit_input_audit.json');fitrows=records('fit_representatives');fitcheck={}
    for n,cols in [('L',[0]),('L+S',[0,1,2,3])]:
        model=cl['models'][n];rates=[];below=[];counts=[]
        for row in audit['rows']:
            path=ROOT/row['feature_path'];assert digest(path)==row['feature_sha256']
            x=np.load(path,mmap_mode='r')[:,cols];truth=np.load(PREP/fitrows[row['product']]['mask_path']);yy=truth[truth!=255]==2
            neg=calc(x,model)[~yy];rates.append(float((neg>=model['threshold']).mean()))
            lower=neg[neg<model['threshold']];below.append(float(lower.max()) if len(lower) else -np.inf)
            counts.append(dict(product=row['product'],negative=len(neg),false_positive=int((neg>=model['threshold']).sum())))
        macro=float(np.mean(rates));assert abs(macro-model['fit_macro_fpr'])<1e-12 and macro<=.01+1e-10
        candidate=max(below);candidate_rates=[]
        if np.isfinite(candidate):
            for row in audit['rows']:
                x=np.load(ROOT/row['feature_path'],mmap_mode='r')[:,cols]
                truth=np.load(PREP/fitrows[row['product']]['mask_path']);yy=truth[truth!=255]==2
                neg=calc(x,model)[~yy];candidate_rates.append(float((neg>=candidate).mean()))
            assert np.mean(candidate_rates)>.01,'Threshold could be lowered without breaking frozen budget'
        fitcheck[n]=dict(macro_fpr=macro,threshold=model['threshold'],next_lower_distinct_score=float(candidate) if np.isfinite(candidate) else None,
            next_lower_macro_fpr=float(np.mean(candidate_rates)) if candidate_rates else None,rows=counts)
    capacity()
    write(ROOT/'SHARED_PIXEL_VERIFIED.json',dict(status='verified_no_new_inference_or_fitting',
        per_scene_AP_independent_tie_endpoint_implementation=checks,fit_full_negative_operating_points=fitcheck,
        native_network_cache_retained_shared=True,local_pixel_cache_recomputation=False,
        code_sha256=digest(Path(__file__)),independent_ap_code_sha256=digest(Path(__file__).with_name('verify_phase78_local.py'))))

if __name__=='__main__':main()
