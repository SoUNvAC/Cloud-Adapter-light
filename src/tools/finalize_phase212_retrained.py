"""Summarize audited Phase212-R data and interpretation in separate artifacts."""
from pathlib import Path
import csv
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'outputs/phase212/phase212_retrain_20261020';OUT=BASE/'evaluation';RAW=OUT/'delivery';SUMMARY=ROOT/'src/research_results/phase212_retrain_20261020'
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def readcsv(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def fmt(x):return '—' if x is None or x=='' else f'{float(x):.4f}'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert load(OUT/'ANALYSIS_ACCEPTANCE.json')['status']=='passed'
    remote=load(RAW/'REMOTE_CPU_ACCEPTANCE.json');assert remote['status']=='passed_CPU_only'
    table=readcsv(OUT/'MAIN_TABLE.csv');relations=readcsv(OUT/'HISTORICAL_POINT_COMPARISONS.csv')
    train=load(RAW/'training_receipts/RETRAINING_COMPLETE.json');ev=load(RAW/'COMPLETED.json');end=load(RAW/'ENDPOINT_REPRODUCTION.json')
    assert train['calls']==15784 and ev['calls'] in [9680,11188]
    lines=['| 目标 | α／历史方法 | 目标mIoU | 源mIoU | 目标增益pp | 源损失pp | 目标Shadow IoU | 源Shadow IoU | 有限集合状态 |','|---|---|---:|---:|---:|---:|---:|---:|---|']
    for x in table:
        status='支配于已列精确点' if x['finite_set_status'].startswith('dominated') else '待历史精确值' if x['finite_set_status'].startswith('pending') else '精确子集非支配*'
        lines.append('| '+x['target']+' | '+x['alpha_or_method'].replace('_seed64_history','（历史）')+' | '+' | '.join(fmt(x[k]) for k in ['target_pooled_mIoU','source_pooled_mIoU','target_gain_vs_Source_pp','source_loss_vs_Source_pp','target_Shadow_IoU','source_Shadow_IoU'])+' | '+status+' |')
    totalcalls=train['calls']+ev['calls'];sec=train['cumulative_execution_seconds']+ev['elapsed_seconds']
    data=f'''## 1．数据结果

用户授权重训版本；两个头各seed64／4000步，按原target-val选mIoU。Source SHA仍为64dd9a…859dc9；L8选3500步、SPARCS选3000步，**新头不是旧SHA原件**。381个Source checkpoint张量初始化精确匹配，281个非head参数／buffer训练前后不变。六个内部头经独立CPU float64 lerp逐项复核，零训练的插值评价均EXIT0。

固定RGB／512／三父类；源535张／107 ROI，L8目标963输入／8scene，SPARCS目标10输入／10scene，均原val，不读test。α同时在源／目标开启。IoU/mIoU单位%；差值pp；下表数字仅展示四位小数，原值在MAIN_TABLE.csv。

'''+ '\n'.join(lines)+f'''

*非支配只针对现有精确坐标子集，不是理论前沿；LoRA历史目标值只有两位小数，完整有限集合状态保留待核验。MsRE/LoRA为历史单seed参照，未重跑，未用关闭恢复成绩或三seed均值。

Source主评价FP32为{end['Source_FP32']['source']:.6f}／{end['Source_FP32']['l8']:.6f}／{end['Source_FP32']['sparcs']:.6f}（源／L8／SPARCS）。历史FP16补验三池预测SHA及混淆精确匹配；主结果仍FP32，历史方法跨精度比较仅描述。重训α=1不能要求等于旧目标成绩。

训练15784次前向；评价{ev['calls']}次（主9680＋精度诊断{ev['calls']-9680}），累计{totalcalls}；阶段墙钟约{sec/60:.2f}分钟，非纯GPU kernel计时。仅保存头、混淆／小回执，不保存完整重复backbone、optimizer或logits。六个头SHA、完整混淆／逐组指标／错误流和24组配对差区间均已交付；bootstrap10000次、seed212020，源按107 ROI、目标按原scene汇总混淆后重算池化mIoU。
'''
    discoveries=[]
    for target in ['l8','sparcs']:
        pts=[x for x in table if x['target']==target and x['alpha_or_method'] in ['0.25','0.5','0.75']]
        gains=[float(x['target_gain_vs_Source_pp']) for x in pts];loss=[float(x['source_loss_vs_Source_pp']) for x in pts]
        discoveries.append(f"{target.upper()}三个内部点的目标增益范围{min(gains):+.4f}至{max(gains):+.4f}pp，源损失范围{min(loss):+.4f}至{max(loss):+.4f}pp。")
    dominates=[f"{x['target'].upper()} α={x['alpha']}" for x in relations if x['historical_method']=='MsRE' and x['relation'].startswith('interpolation point-estimate dominates')]
    uncovered=[t.upper() for t in ['l8','sparcs'] if not any(x['target']==t and x['historical_method']=='MsRE' and x['relation'].startswith('interpolation point-estimate dominates') for x in relations)]
    comparison=('在已封存精确历史坐标中，'+ '、'.join(dominates)+'对MsRE形成描述性的点估计支配。') if dominates else '没有内部点对已封存MsRE精确历史坐标形成点估计支配。'
    if uncovered:comparison+=' '+ '、'.join(uncovered)+'的MsRE坐标未被本轮三个内部点覆盖，不能说该方法已被普遍替代。'
    conclusions='''## 2．结论分析

1. **简单头部插值的作用。** '''+' '.join(discoveries)+''' 这给出了有限固定点上的保源／目标取舍；是否“有用”应结合保留的源损失和目标收益判断，不把位于端点之间或超过连线当作机制证明。
2. **既有方法是否仍有价值。** '''+comparison+''' LoRA的目标历史值精度不足，支配关系明确保留“待精确值核验”。新点为FP32、历史方法为FP16且未重跑，不能把上述描述升级为公平精度控制后的普遍胜负或显著二维支配。
3. **下一步是否值得做。** 值得将这种已知WiSE-FT式头部权重插值保留为低成本基线；是否投入新机制，需先证明其超出该简单基线的价值。本轮只复用了一个seed和目标val选择的端点，源val也已用于探索；区间条件于固定checkpoint，不覆盖训练／选模不确定性。未做标签效率曲线、独立确认、通用DINO能力保持或因果机制验证，也没有形成新算法。
'''
    (OUT/'DATA_RESULTS.md').write_text(data,encoding='utf-8');(OUT/'CONCLUSIONS.md').write_text(conclusions,encoding='utf-8');(OUT/'RESULT.md').write_text('# Phase212-R：VFM头部插值保源探针\n\n'+data+'\n'+conclusions,encoding='utf-8')
    cost={'status':'completed_retraining_and_fixed_interpolation','training_calls':train['calls'],'evaluation_calls':ev['calls'],'main_evaluation_calls':9680,'FP16_diagnostic_calls':ev['calls']-9680,'failed_model_calls':0,'all_model_calls':totalcalls,'training_cumulative_wall_seconds':train['cumulative_execution_seconds'],'evaluation_wall_seconds':ev['elapsed_seconds'],'combined_stage_wall_seconds':sec,'new_combined_persistent_bytes_before_delivery':ev['new_combined_persistent_bytes'],'limits':{'training_calls':16000,'evaluation_calls':16000,'training_stage_seconds':14400,'evaluation_stage_seconds':14400,'persistent_bytes':500000000,'CPU_bytes':16*1024**3,'GPU_bytes':20*1024**3},'resource_limitations':['evaluation CPU peak not continuously sampled; unknown','training runtime includes1.3337932010181248s zero-call missing-psutil failure','stage wall time includes input/model setup; not pure GPU kernel time','host fixed-disk free was sampled at preflight; no VHD compact or history cleanup']}
    (OUT/'COST_AND_LIMITATIONS.json').write_text(json.dumps(cost,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    readme='''# Phase212-R\n\n已完成两个头的节省存储重训和固定α插值评价。数据结果见DATA_RESULTS.md，结论探讨见CONCLUSIONS.md，合并两页报告RESULT.md。\n\nMAIN_TABLE.csv为主表；ALL_POOL_METRICS.csv、PER_GROUP.csv、ERROR_FLOWS.csv保留完整三类、召回、FPR／最大组误报和混淆。PAIRED_DIFFERENCES_CI.csv为按ROI／scene整组配对bootstrap区间。HISTORICAL_POINT_COMPARISONS.csv保留历史精度不足项，不能包装成独立确认。\n\n原Phase212缺件门检报告保留不改；本轮是用户授权的新重训版本，不是恢复旧目标SHA。新头与六个插值头留在shared，未下载重复权重、同步work_dirs、自动push、建定时任务或启动后续phase。\n'''
    (OUT/'README.md').write_text(readme,encoding='utf-8')
    SUMMARY.mkdir(exist_ok=True)
    for name in ['DATA_RESULTS.md','CONCLUSIONS.md','RESULT.md','README.md','MAIN_TABLE.csv','ALL_POOL_METRICS.csv','PAIRED_DIFFERENCES_CI.csv','HISTORICAL_POINT_COMPARISONS.csv','ANALYSIS_ACCEPTANCE.json','COST_AND_LIMITATIONS.json']:
        shutil.copyfile(OUT/name,SUMMARY/name)
    hashes={str(p.relative_to(OUT)).replace('\\','/'):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in OUT.rglob('*') if p.is_file() and p.name!='DELIVERY_COMPLETE.json'}
    receipt={'status':'complete','retraining_and_fixed_evaluation_complete':True,'new_original_SHA_recovery_claim':False,'zero_test_access':True,'files':hashes,'cost':cost,'no_new_phase_or_timer_or_Git_push':True}
    (OUT/'DELIVERY_COMPLETE.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');shutil.copyfile(OUT/'DELIVERY_COMPLETE.json',SUMMARY/'DELIVERY_COMPLETE.json')
    (BASE/'HANDOFF.md').write_text('# Phase212-R handoff\n\nCurrent delivery: COMPLETE. Two retrained best heads and six fixed interpolation heads sealed on shared; no further fits/forwards. Training15784 calls; evaluation'+str(ev['calls'])+' calls. Source historical FP16 prediction SHA/confusion reproduced exactly, main fixed FP32 retained. All source/head/buffer/frozen, independentCPU confusion and paired bootstrap checks passed. See evaluation/RESULT.md and DELIVERY_COMPLETE.json. No timer created, no push or next phase. Original blocked preflight unchanged.\n',encoding='utf-8')
    print(json.dumps({'status':'complete','report':str(OUT/'RESULT.md'),'dominates_MsRE_points':dominates,'calls':totalcalls},ensure_ascii=False))

if __name__=='__main__':main()
