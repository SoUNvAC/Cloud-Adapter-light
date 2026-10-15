"""Validate fixed endpoints and old baselines, publish small tables, no weights in Git."""
import csv,json,hashlib,struct,shutil
from pathlib import Path
from evaluate_phase123_local import R,U,O,load,sha,save
def main():
    assert load(O/'LOCAL_CPU_ACCEPTANCE.json')['status']=='passed'
    d=U/'delivery';training=load(d/'train_C/TRAINING_RECEIPT.json');control=load(d/'CONTROLLER_STATUS.json')
    assert training['steps']==1000 and training['seed']==12301 and training['encoder_unchanged'] and training['decoder_updated']
    assert training['final_model_sha256']==sha(U/'C_decoder_final.safetensors')
    assert control['calls_completed']==2392
    import numpy as np
    raw=(U/'C_decoder_final.safetensors').read_bytes();size=struct.unpack('<Q',raw[:8])[0];header=json.loads(raw[8:8+size]);count=0;end=0
    for name,item in sorted(((k,v) for k,v in header.items() if k!='__metadata__'),key=lambda kv:kv[1]['data_offsets'][0]):
        a,b=item['data_offsets'];assert a==end;end=b
        v=np.frombuffer(raw[8+size+a:8+size+b],dtype={'F32':'<f4','I64':'<i8'}[item['dtype']]).reshape(item['shape'])
        assert np.isfinite(v).all();count+=1
    assert count==61 and end==len(raw)-8-size
    costs=[];scenes=[]
    for stage in ('train_C','C_Calibration','C_Catalogue','C_PixBox'):
        status=load(d/stage/'RUN_STATUS.json');costs.append(dict(stage=stage,**status))
        if stage!='train_C':
            info=load(d/stage/'MODEL_INFO.json');assert info['models']==1 and info['total_parameters']==18039132 and info['deployed_teacher_models']==0
            save(O/(stage+'_SCORES_SEALED.json'),load(d/stage/'SCORES_SEALED.json'))
            for v in load(d/stage/'SCENE_RECEIPTS.json'):
                assert v['coverage_gradient_exact'] and v['max_confidence_error']<=1e-5 and v['max_raw_mean_error']<=1e-5
                scenes.append(v)
    assert sum(v['all_calls'] for v in costs)==2392 and len(scenes)==101
    save(O/'ACTUAL_RESOURCE_COST.json',dict(model_calls=2392,controller_wall_seconds=control['elapsed_seconds'],stages=costs,
        max_independent_confidence_error=max(v['max_confidence_error'] for v in scenes),validated_scenes=len(scenes)))
    for pool in ('Catalogue','PixBox'):
        assert load(O/(pool+'_CPU_ACCEPTANCE.json'))['support_loss']==0,'resolve support mismatch before reusing old means'
        with (O/(pool+'_six_workpoints.csv')).open(encoding='utf-8',newline='') as f:new=list(csv.DictReader(f))
        with (R/'src/research_results/phase122_20261015'/(pool+'_four_workpoints.csv')).open(encoding='utf-8',newline='') as f:old=list(csv.DictReader(f))
        for row in old:
            name=row['setting'].replace('C100','C122');actual=next(v for v in new if v['setting']==name)
            for key in ('macro_AP_pp','pooled_AP_pp','macro_recall_pp','macro_FPR_pp'):
                assert float(actual[key])==float(row[key]),(pool,name,key)
    for name in ['TRAINING_SEAL.json','THRESHOLD_LOCK.json','ALL_SCORES_SEALED.json','CALIBRATION_INDEPENDENT_CPU_ACCEPTANCE.json','CALIBRATION_SCORE_LOCATIONS.json','FIT_INPUT_LOCK.json','PAIRED_CROP_SCHEDULE.json','CONTROLLER_STATUS.json']:
        save(O/name,load(d/name))
    for name in ['TRAINING_RECEIPT.json','PARAMETER_WHITELIST.json','INITIAL_GRADIENT_GATE.json','RUN_STATUS.json']:
        save(O/('C123_'+name),load(d/'train_C'/name))
    save(O/'COMPLETION.json',dict(status='completed_exploratory',phase=123,seed=12301,training_steps=1000,training_groups=100,calibration_groups=53,
        model_calls=2392,model_retries=0,decoder_sha256=sha(U/'C_decoder_final.safetensors'),new_data_downloads=0,reference_uploads=0,work_dirs_transfers=0,
        baseline_reproduction='both R and seed12201, four workpoints, AP/pooledAP/recall/FPR exact',new_independent_confirmation=False))
    lines=['# Phase123 数据结果','', '一个新C100，seed12301，固定100组/1000步；53组只校准本轮封存模型。R及seed12201分数/阈值不变。未生成分数集成。', '',
        '| 池 | 工作点 | 宏AP % | 池化AP % | 宏召回 % | 组均FPR % | FPR单侧95%上界 % | 最大组FPR % |','|---|---|---:|---:|---:|---:|---:|---:|']
    for pool in ('Catalogue','PixBox'):
        with (O/(pool+'_six_workpoints.csv')).open(encoding='utf-8',newline='') as f:
            for v in csv.DictReader(f):lines.append('| '+pool+' | '+v['setting']+' | '+' | '.join(f'{float(v[k]):.4f}' for k in ('macro_AP_pp','pooled_AP_pp','macro_recall_pp','macro_FPR_pp','FPR_U95_pp','max_group_FPR_pp'))+' |')
        m=load(O/(pool+'_comparisons.json'))['primary'];lines+=['',pool+'两seed先按组平均后相对固定R：召回差 '+str(m['recall'])+'；FPR差 '+str(m['FPR'])+'；AP差 '+str(m['AP'])+'。','']
    lines+=['Catalogue32组/7正类组；PixBox16产品/10组/8正类组，误报只针对选定负参考点。区间为10000次相关组bootstrap，seed123015，条件于本轮两seed，不是两倍独立组，也不是跨seed总体置信结论。',
        '组均1%是校准经验目标，不是评价FPR相同或逐景保证。历史暴露、未知预训练交集及稀疏参考限制保留。完整逐组/Surface/Cloud/零召回/超预算见CSV。结论另写ANALYSIS.md；不自动启动124。']
    with (O/'README.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    for p in O.rglob('*'):
        if p.is_file():
            target=U/p.relative_to(O);target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():assert sha(p)==sha(target),str(target)
            else:shutil.copyfile(p,target)
    print('Phase123 final data verified; write ANALYSIS/NEXT_STEP and publish small artifacts only')
if __name__=='__main__':main()
