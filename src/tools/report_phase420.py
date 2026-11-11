"""Independent CPU audit of saved hard maps, frozen baseline, paired data and AP."""
from pathlib import Path
import json,csv,hashlib,time,sys
import numpy as np
R=Path('/home/scv/shared/phase420_20261101')
REF=Path('/home/scv/shared/phase411_20261101/layerwise')
def js(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,v):(R/n).write_text(json.dumps(v,indent=2))
def write(n,rows):
    with (R/n).open('w',newline='') as f:
        w=csv.DictWriter(f,list(dict.fromkeys(k for row in rows for k in row)));w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()})
def metric(cm):
    d=np.diag(cm);u=cm.sum(0)+cm.sum(1)-d;ii=np.divide(100*d,u,out=np.full(3,np.nan),where=u!=0)
    return {'mIoU':float(np.nanmean(ii)),'IoU':[None if np.isnan(v) else float(v) for v in ii],'Shadow_IoU':None if np.isnan(ii[2]) else float(ii[2]),'confusion':cm.tolist()}
def ap(s,y):
    P=int(y.sum())
    if P==0 or P==len(y):return None
    order=np.argsort(-s,kind='stable');s=s[order];y=y[order];ends=np.r_[np.flatnonzero(s[1:]!=s[:-1]),len(s)-1];tp=np.cumsum(y,dtype=np.int64)[ends]
    return float(np.sum(np.diff(np.r_[0,tp])*tp/(ends+1))/P)
def main():
    start=time.monotonic();done=js(R/'COMPLETE.json');assert done['passed']
    for n,h in js(R/'LOCK.json')['files'].items():assert sha(R/n)==h
    baseline=js(REF/'evaluations/M/step_1000/METRICS.json');bm={g['product']:g for g in baseline['per_group']}
    refs=[json.loads(s) for s in (REF/'work_dirs/formal/M/STEP_TRACE.jsonl').read_text().splitlines()][:1000]
    ledger=[json.loads(s) for s in (R/'CALL_LEDGER.jsonl').read_text().splitlines()];assert len(ledger)==2192
    assert sum(r['stage']=='train' for r in ledger)==2000
    summary=[];groups=[];costs=[]
    for mode in ['global','spatial']:
        scope=js(R/mode/'SCOPE.json');assert scope['trainable_parameters']==6817
        trace=[json.loads(s) for s in (R/mode/'STEP_TRACE.jsonl').read_text().splitlines()];assert len(trace)==1000
        for x,y in zip(trace,refs):
            assert all(x[k]==y[k] for k in ['step','product','image_SHA','label_SHA','augmentation_seed'])
            assert x['LR']==y['LR'][0] and np.isfinite(x['loss']) and np.isfinite(x['gradient_norm'])
        for step in [1]+list(range(100,1001,100)):
            frozen=js(R/mode/f'FREEZE_{step}.json');assert frozen['passed'] and frozen['state_SHA']==scope['baseline_frozen_SHA']
        for step in [0,500,1000]:
            rep=js(R/mode/f'evaluations/step_{step}/METRICS.json');assert len(rep['per_group'])==32
            total=np.zeros((3,3),np.int64);aps=[];scores=[];truths=[];fixes=0;regressions=0
            for g in rep['per_group']:
                p=R/g['score_file'];assert sha(p)==g['score_SHA']
                with np.load(p,allow_pickle=False) as z:
                    pred=z['pred'];b=z['baseline_pred'];truth=z['truth'];valid=truth!=255;s=z['shadow'][valid].astype(np.float64);y=truth[valid]==2
                    assert np.isfinite(s).all() and s.min()>=0 and s.max()<=1
                    assert np.isin(pred,[0,1,2]).all() and np.isin(b,[0,1,2]).all()
                    cm=np.bincount(3*truth[valid].astype(np.int64)+pred[valid],minlength=9).reshape(3,3)
                    assert cm.tolist()==g['confusion']
                    bcm=np.bincount(3*truth[valid].astype(np.int64)+b[valid],minlength=9).reshape(3,3)
                    assert bcm.tolist()==bm[g['product']]['confusion']
                    assert hashlib.sha256(b.tobytes()).hexdigest()==bm[g['product']]['prediction_SHA']
                    assert not np.any(((b==0)&(pred==1))|((b==1)&(pred==0)))
                    if step==0:assert np.array_equal(b,pred)
                    f=int(np.sum(valid&(pred==truth)&(b!=truth)));e=int(np.sum(valid&(pred!=truth)&(b==truth)))
                    assert f==g['fixes'] and e==g['regressions'];fixes+=f;regressions+=e
                a=ap(s,y);aps.append(a);scores.append(s);truths.append(y);total+=cm
                groups.append({'mode':mode,'step':step,'product':g['product'],**metric(cm),'AP':None if a is None else 100*a,'fixes':f,'regressions':e,'forbidden_surface_cloud_switches':0})
            assert total.tolist()==rep['confusion'] and abs(metric(total)['mIoU']-rep['mIoU'])<1e-9
            assert fixes==rep['fixes'] and regressions==rep['regressions'] and sum(a is not None for a in aps)==7
            summary.append({'mode':mode,'step':step,**metric(total),'macro_AP':100*float(np.mean([a for a in aps if a is not None])),'pooled_AP':100*ap(np.concatenate(scores),np.concatenate(truths)),'fixes':fixes,'regressions':regressions,'forbidden_surface_cloud_switches':0})
            save('REPORT_STATUS.json',{'mode':mode,'step':step,'seconds':time.monotonic()-start})
        import torch
        ck=torch.load(R/mode/'iter_1000.pth',map_location='cpu')
        assert ck['step']==1000 and ck['base_checkpoint_SHA']==done['frozen_baseline_checkpoint_SHA'] and sum(v.numel() for v in ck['head'].values())==6817
        assert all(torch.isfinite(v).all() for v in ck['head'].values())
        costs.append({'mode':mode,**{k:done['reports'][mode][k] for k in ['parameters','calls','updates','seconds']}})
    assert js(R/'global/SCOPE.json')['head_initial_SHA']==js(R/'spatial/SCOPE.json')['head_initial_SHA']
    write('TARGET_SUMMARY.csv',summary);write('TARGET_GROUPS.csv',groups);write('COST.csv',costs)
    endpoint={r['mode']:r for r in summary if r['step']==1000};margin=endpoint['spatial']['mIoU']-max(baseline['mIoU'],endpoint['global']['mIoU'])
    save('MECHANISM_SCREEN.json',{'margin_pp':margin,'clears_0_30pp':margin>=.3,'baseline_mIoU':baseline['mIoU'],'comparison':'spatial exceeds frozen baseline and image-global control by >=0.30pp at step1000','automatic_promotion':False})
    accepted={'passed':True,'new_calls':2192,'new_updates':2000,'parameters_per_arm':6817,'baseline_hard_predictions_exact_all_evaluations':True,'baseline_frozen':True,'input_augmentation_LR_paired':True,'hard_map_confusion_recomputed':True,'forbidden_surface_cloud_switches':0,'fix_regression_counts_recomputed':True,'exact_AP_recomputed':True,'CPU_report_seconds':time.monotonic()-start}
    save('INDEPENDENT_ACCEPTANCE.json',accepted)
    lines=['# Phase420：固定非阴影比例的阴影修正','',f"冻结Phase411第1000步目标基线：{baseline['mIoU']:.4f}% mIoU。153 fit、32 development、seed40901；两组各追加1000步，不能视为与从头1000步相同训练预算。标签及场景全部保留。",'','|方案|第1000步mIoU|影IoU|影宏AP|修复像素|新增错误像素|','|---|---:|---:|---:|---:|---:|']
    for mode,r in endpoint.items():lines.append(f"|{mode}|{r['mIoU']:.4f}|{r['Shadow_IoU']:.4f}|{r['macro_AP']:.4f}|{r['fixes']}|{r['regressions']}|")
    lines+=['',f'空间分支相对较强基线/整图对照余量：{margin:+.4f}个百分点；达到预设0.30点门槛：{margin>=.3}。未自动延长训练。','','两组使用相同6817参数分支、输入与每图类别等权NLL；global把输出空间平均，spatial逐像素输出。结构输入包括冻结特征、三类概率、RGB亮度、局部亮度差和梯度；只是最小结构验证，尚不证明云影专用性或新颖性。','','约束在原图尺度执行：保留云/地表条件比例，只调整阴影权重。数学检查、全部原图预测逐像素检查通过，地表↔云直接切换为0；仍允许阴影误报/漏检，不能保证总体提升。','','每组0/500/1000步全量评价，固定1000步为主结果。基线预测每次均与封存结果逐像素哈希一致；基线权重冻结。完整硬预测、基线预测、真值和阴影得分保存在远端，逐场景修复/新增错误见TARGET_GROUPS.csv。','',f"新增2192次完整输入前向、2000次更新，GPU阶段累计{sum(c['seconds'] for c in costs)/60:.2f}分钟（包含0步复现评价）。单seed已暴露development结果，不作稳定性结论。"]
    (R/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save('REPORT_COMPLETE.json',{'passed':True,'files':{n:sha(R/n) for n in ['RESULT.md','TARGET_SUMMARY.csv','TARGET_GROUPS.csv','COST.csv','MECHANISM_SCREEN.json','INDEPENDENT_ACCEPTANCE.json']}})
if __name__=='__main__':
    import traceback
    try:main()
    except BaseException:save('REPORT_FAILURE.json',{'traceback':traceback.format_exc()});raise
