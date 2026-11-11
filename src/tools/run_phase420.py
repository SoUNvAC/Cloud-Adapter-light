"""Two-stage frozen target model plus constrained shadow correction; no label edits."""
from pathlib import Path
import os,sys,json,hashlib,time,traceback,copy,shutil
R=Path('/home/scv/shared/phase420_20261101')
REF=Path('/home/scv/shared/phase411_20261101/layerwise')
BASE=REF/'work_dirs/formal/M/best_mIoU_iter_1000.pth'
os.environ['PHASE400_ROOT']=str(R)
sys.path.insert(0,str(R/'code'))
START=time.monotonic()
def js(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()
def save(n,v):
    p=R/n;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False));tmp.replace(p)
def main():
    lock=js(R/'LOCK.json')
    for n,h in lock['files'].items():assert sha(R/n)==h,n
    assert shutil.disk_usage(R).free>5*1024**3
    for n,h in js(R/'BASE_CODE_LOCK.json').items():assert sha(R/'code'/n)==h
    from phase400_common import setup,state_sha,tensor_sha
    torch=setup();torch.manual_seed(40901)
    import numpy as np
    import torch.nn.functional as F
    from phase420_shadow import ShadowCorrection,normalize,correct,self_test
    save('SELF_TEST.json',self_test())
    from mmengine.config import Config
    from mmengine.dataset import pseudo_collate
    from mmseg.registry import MODELS,DATASETS
    from mmseg.structures import SegDataSample
    import phase400_runtime
    cfg=Config.fromfile(str(REF/'configs/formal_M.json'))
    # Serialized JSON loses tuple types required by crop/resize transforms.
    for loader in [cfg.train_dataloader,cfg.val_dataloader]:
        for transform in loader.dataset.pipeline:
            for key in ['crop_size','scale']:
                if key in transform and isinstance(transform[key],list):transform[key]=tuple(transform[key])
    model=MODELS.build(cfg.model).cuda();checkpoint=torch.load(BASE,map_location='cpu')
    assert checkpoint['phase400']['draw_cursor']==1000
    model.load_state_dict(checkpoint['state_dict'],strict=True);del checkpoint
    model.eval();model.requires_grad_(False);frozen=state_sha(model.state_dict());base_sha=sha(BASE)
    save('BASE_LOCK.json',{'checkpoint':str(BASE),'SHA':base_sha,'state_SHA':frozen,'stage':'already trained Phase411 target model, step1000'})
    train_cfg=copy.deepcopy(cfg.train_dataloader.dataset);train_cfg.prepared_manifest=str(R/'PREPARED_MANIFEST.json')
    train=DATASETS.build(train_cfg)
    val_cfg=copy.deepcopy(cfg.val_dataloader.dataset);val_cfg.prepared_manifest=str(R/'PREPARED_MANIFEST.json')
    val=DATASETS.build(val_cfg);assert len(train)==153 and len(val)==32
    trace=[json.loads(s) for s in (REF/'work_dirs/formal/M/STEP_TRACE.jsonl').read_text().splitlines()][:1000]
    lookup={Path(train.get_data_info(i)['img_path']).name:i for i in range(len(train))}
    expected=js(REF/'evaluations/M/step_1000/METRICS.json');expected_map={g['product']:g for g in expected['per_group']}
    ledger=(R/'CALL_LEDGER.jsonl').open('x',buffering=1);calls=0
    def forward_base(sample,mode,stage,step):
        nonlocal calls
        data=model.data_preprocessor(pseudo_collate([sample]),training=stage=='train')
        with torch.no_grad():
            features=model.extract_feat(data['inputs'])
            scores=model.decode_head.predict(features,[s.metainfo for s in data['data_samples']],model.test_cfg)
        assert scores.shape[-2:]==data['inputs'].shape[-2:]
        calls+=1;ledger.write(json.dumps({'ordinal':calls,'mode':mode,'stage':stage,'step':step,'product':Path(sample['data_samples'].img_path).name})+'\n')
        prob=normalize(scores)
        rgb=((data['inputs']*model.data_preprocessor.std+model.data_preprocessor.mean)/255.).clamp(0,1)
        return data,features[-1],scores,prob,rgb
    def evaluate(head,mode,step):
        head.eval();out=R/mode/f'evaluations/step_{step}';out.mkdir(parents=True)
        total=np.zeros((3,3),np.int64);base_total=total.copy();groups=[];ratio_max=0.;forbidden=0;fixes=0;regressions=0
        with torch.inference_mode():
            for index in range(len(val)):
                sample=val[index];data,feat,scores,p,rgb=forward_base(sample,mode,'development',step)
                delta=head(feat,p,rgb)
                metas=[SegDataSample(metainfo=s.metainfo) for s in data['data_samples']]
                original=model.postprocess_result(scores,data['data_samples'])[0]
                delta_original=model.postprocess_result(delta.expand(-1,3,-1,-1),metas)[0].seg_logits.data[:1][None]
                original_p=normalize(original.seg_logits.data[None]);q=correct(original_p,delta_original)
                pred=q.argmax(1)[0];bp=original.pred_sem_seg.data[0]
                ratio=float((q[:,0]*original_p[:,1]-q[:,1]*original_p[:,0]).abs().max());ratio_max=max(ratio_max,ratio);assert ratio<1e-6
                bad=int((((bp==0)&(pred==1))|((bp==1)&(pred==0))).sum());assert bad==0;forbidden+=bad
                truth=sample['data_samples'].gt_sem_seg.data[0].detach().cpu().numpy().astype(np.uint8)
                b=bp.cpu().numpy().astype(np.uint8);y=pred.cpu().numpy().astype(np.uint8);valid=truth!=255
                cm=np.bincount(3*truth[valid].astype(np.int64)+y[valid],minlength=9).reshape(3,3)
                bcm=np.bincount(3*truth[valid].astype(np.int64)+b[valid],minlength=9).reshape(3,3)
                product=Path(sample['data_samples'].img_path).name.removesuffix('_rgb.npy');ref=expected_map[product]
                assert bcm.tolist()==ref['confusion'],('baseline reproduction CM',product)
                assert hashlib.sha256(b.tobytes()).hexdigest()==ref['prediction_SHA'],('baseline prediction hash',product)
                if step==0:assert np.array_equal(y,b)
                f=int(np.sum(valid&(y==truth)&(b!=truth)));e=int(np.sum(valid&(y!=truth)&(b==truth)));fixes+=f;regressions+=e
                path=out/(product+'.npz');np.savez(path,shadow=q[0,2].cpu().numpy(),truth=truth,pred=y,baseline_pred=b)
                groups.append({'product':product,'group_id':ref['group_id'],'confusion':cm.tolist(),'baseline_confusion':bcm.tolist(),'fixes':f,'regressions':e,'forbidden_surface_cloud_switches':bad,'score_file':str(path.relative_to(R)),'score_SHA':sha(path),'prediction_SHA':hashlib.sha256(y.tobytes()).hexdigest()})
                total+=cm;base_total+=bcm
        d=np.diag(total);union=total.sum(0)+total.sum(1)-d;ii=(100*d/union).tolist();assert base_total.tolist()==expected['confusion']
        report={'mode':mode,'step':step,'mIoU':sum(ii)/3,'IoU':ii,'confusion':total.tolist(),'per_group':groups,'baseline_prediction_reproduced':True,'ratio_crossproduct_max_error':ratio_max,'forbidden_surface_cloud_switches':forbidden,'fixes':fixes,'regressions':regressions}
        save(f'{mode}/evaluations/step_{step}/METRICS.json',report);head.train();return report
    results={};init_sha=None
    for mode in ['global','spatial']:
        begin=time.monotonic();torch.manual_seed(4200101);head=ShadowCorrection(mode).cuda();assert sum(p.numel() for p in head.parameters())==6817
        now=state_sha(head.state_dict());init_sha=now if init_sha is None else init_sha;assert now==init_sha
        opt=torch.optim.AdamW(head.parameters(),lr=1e-4,weight_decay=.05,eps=1e-8,betas=(.9,.999))
        save(mode+'/SCOPE.json',{'trainable_parameters':6817,'head_initial_SHA':now,'baseline_frozen_SHA':frozen,'loss':'mean NLL within each observed GT class, then mean across observed classes','additional_steps':1000})
        save('STATUS.json',{'mode':mode,'step':0,'stage':'identity_evaluation','seconds':time.monotonic()-START})
        evaluate(head,mode,0);records=[];startcalls=calls-32
        with (R/mode/'STEP_TRACE.jsonl').open('x',buffering=1) as stream:
            for ref in trace:
                step=ref['step'];sample=train[(lookup[ref['product']],step-1)]
                assert tensor_sha(sample['inputs'])==ref['image_SHA']
                assert tensor_sha(sample['data_samples'].gt_sem_seg.data)==ref['label_SHA']
                assert sample['data_samples'].metainfo['phase400_augmentation_seed']==ref['augmentation_seed']
                data,feat,scores,p,rgb=forward_base(sample,mode,'train',step)
                opt.zero_grad(set_to_none=True);delta=head(feat,p,rgb);q=correct(p,delta)
                labels=data['data_samples'][0].gt_sem_seg.data[0].long();losses=[]
                for cls in range(3):
                    mask=labels==cls
                    if mask.any():losses.append(-q[0,cls][mask].clamp_min(1e-8).log().mean())
                assert losses;loss=torch.stack(losses).mean();assert torch.isfinite(loss)
                loss.backward();assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in head.parameters())
                norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
                lr=ref['LR'][0];assert len(set(ref['LR']))==1
                for group in opt.param_groups:group['lr']=lr
                opt.step();torch.cuda.synchronize()
                stream.write(json.dumps({'step':step,'product':ref['product'],'image_SHA':ref['image_SHA'],'label_SHA':ref['label_SHA'],'augmentation_seed':ref['augmentation_seed'],'LR':lr,'loss':float(loss),'gradient_norm':float(norm)})+'\n')
                if step==1 or step%100==0:
                    assert state_sha(model.state_dict())==frozen
                    assert all(v.grad is None and not v.requires_grad for v in model.parameters())
                    save(mode+f'/FREEZE_{step}.json',{'passed':True,'state_SHA':frozen,'step':step})
                    save('STATUS.json',{'mode':mode,'step':step,'stage':'train','seconds':time.monotonic()-START})
                    assert time.monotonic()-START<7200
                if step%500==0:
                    records.append(evaluate(head,mode,step))
                    torch.save({'head':head.state_dict(),'optimizer':opt.state_dict(),'step':step,'mode':mode,'base_checkpoint_SHA':base_sha,'input_lock_SHA':sha(R/'INPUT_LOCK.json')},R/mode/f'iter_{step}.pth')
        assert calls-startcalls==1096
        results[mode]={'parameters':6817,'calls':1096,'updates':1000,'seconds':time.monotonic()-begin,'rows':[{k:v for k,v in row.items() if k!='per_group'} for row in records]}
        save('SCREEN.json',results)
    ledger.close();assert calls==2192
    save('COMPLETE.json',{'passed':True,'reports':results,'new_calls':calls,'new_updates':2000,'frozen_baseline_checkpoint_SHA':base_sha,'seconds':time.monotonic()-START,'scope':'additional branch training on a previously trained target baseline; exposed single-seed development'})
    (R/'EXIT_CODE').write_text('0')
if __name__=='__main__':
    try:main()
    except BaseException:save('FAILURE.json',{'traceback':traceback.format_exc()});(R/'EXIT_CODE').write_text('1');raise
