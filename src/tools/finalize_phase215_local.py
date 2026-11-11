"""Short Phase215 result, true costs and immutable delivery manifest."""
from pathlib import Path
import csv,hashlib,json,shutil,time,ctypes
from ctypes import wintypes
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/phase215/phase215_20261023';SUMMARY=ROOT/'src/research_results/phase215_20261023';START=time.monotonic()
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def rd(n):return list(csv.DictReader((OUT/n).open(encoding='utf-8-sig')))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(OUT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def main():
    acceptance=js(OUT/'INDEPENDENT_CPU_ACCEPTANCE.json');assert acceptance['status']=='passed'
    lock=js(OUT/'SOURCE_SELECTION_LOCK.json');assert sha(OUT/'SOURCE_SELECTION_LOCK.json')==js(OUT/'SOURCE_ONLY_ACCEPTANCE.json')['source_lock_sha256']
    assert lock['selection_code_sha256']==sha(ROOT/'src/tools/select_phase215_source.py')
    registry=js(OUT/'INPUT_REGISTRY.json')
    for stage in ['213','214']:
        base=Path(registry['sources'][stage]['root']);m=js(base/'DELIVERY_MANIFEST.json');assert sha(base/'DELIVERY_MANIFEST.json')==registry['sources'][stage]['manifest_sha256']
        assert all((base/n).stat().st_size==v['bytes'] and sha(base/n)==v['sha256'] for n,v in m.items())
    compare=rd('RULE_COMPARISON.csv');metrics=rd('RULE_METRICS_SIDE_BY_SIDE.csv');changes=rd('SOURCE_LEAVE_ONE_SELECTION.csv')
    def transition(a,b):return f'{float(a):.2f}→{float(b):.2f}'
    table1='\n'.join(f"| {r['target']}/B{r['budget_pp']} | {transition(r['old_alpha'],r['new_alpha'])} | {transition(r['old_test_mIoU'],r['new_test_mIoU'])} | {float(r['new_gain_vs_Source_pp']):+.2f} | {float(r['forgone_test_gain_pp']):+.2f} |" for r in compare)
    table2='\n'.join(f"| {r['target']}/B{r['budget_pp']} | {transition(r['old_source_pool_loss_pp'],r['new_source_pool_loss_pp'])} | {transition(r['old_T11_pp'],r['new_T11_pp'])} | {transition(r['old_worst_ROI_loss_pp'],r['new_worst_ROI_loss_pp'])} | {r['old_ROI_loss_gt5_count']}→{r['new_ROI_loss_gt5_count']} |" for r in compare)
    report='''# Phase215：源域平均＋尾部约束的固定α回放

## 1．数据结果

全本地CPU，零训练／拟合／前向／远程／下载／像元读取，未新增α或bootstrap。复用212-R FP32端点及213/214已交付混淆；Source源mIoU77.8152217929%。先仅解析源107 ROI混淆封存SOURCE_SELECTION_LOCK，再连接既有目标表。原213/214所有封存SHA复核通过。人已看过历史目标结果，本轮是事后探索规则，不是盲测、独立确认或新算法。

规则固定为最大α满足：源池化损失≤B，且最差11/107 ROI带符号损失均值T11≤5pp；B为1/3/5。未舍入比较，无epsilon，未截断负损失。5pp是事后固定容忍度，不是公认业务安全线。

**完整目标支持上的代价。** mIoU为%，增益/放弃为pp；放弃=旧规则−新规则，保留符号。L8完整16景、SPARCS10景。

| 目标/预算 | 旧α→新α | 旧→新test mIoU | 新增益vs Source | 放弃收益 |
|---|---|---|---:|---:|
'''+table1+'''

**对应源域代价。** 每一项均旧→新，损失为pp；B1/B3在L8原为同点，SPARCS新规则三预算均同点，不能算重复验证。

| 目标/预算 | 池化损失 | T11 | 最差ROI损失 | ROI损失>5数 |
|---|---|---|---|---|
'''+table2+'''

L8 α=.25的T11=5.255191436169095，严格超5，所有非零候选均不符合尾约束，故三预算都回退Source。SPARCS只有0和.25符合尾约束，三预算均选.25；仍有ROI_0626损失6.4438pp，T11≤5不等于每ROI≤5。

SPARCS新点相对Source增益+7.7582pp，直接引用213同α的95%CI[5.2134,10.7530]；L8新点为Source，差与区间均0。这些区间条件于固定模型/α，不覆盖事后规则设计或源选择不确定性。新−旧差仅点估计，未用两个区间端点相减拼CI。

指标代价没有只看mIoU：SPARCS B5改选后目标Shadow IoU50.2470→43.4485%，召回61.2890→54.3387%，名义FPR2.9130→3.3225%；源Shadow IoU53.1098→60.5200%。完整源/目标各类IoU、召回、误报并列于RULE_METRICS_SIDE_BY_SIDE.csv。L8 yes/no只引用214辅助表，不据其重选；no不是可靠真实无影，yes也非本轮人工认证。

**源选择留一。** 每次106 ROI，尾部仍11。L8各预算104次选0、3次选.25，变化仅删除ROI_0124、ROI_0626、ROI_0714之一时触发；SPARCS各预算107次均选.25。删除敏感性不是泛化证明，不删除这些ROI改变最终选择。全部10候选可行性、642个留一明细、次数和变化ID均随附。

## 2．验收与范围

独立脚本未导入主函数，以逐类整数union、heapq最大11项及statistics.fmean复核1200组IoU、全部10候选、6条选择与642个留一，次数均合计107。核验严格边界、α=0回退、带符号T11和放弃收益；214引用和原封存未变。既有尾部名单/净错误流按SHA引用，不读取新像元、不猜地物机制。

真实阶段耗时、Windows进程峰值、文件字节数及代码/输入SHA见COST_RECEIPT和交付清单。未做新检验、未挑最佳预算部署、未与旧FP16/不同支持的MsRE或LoRA宣称支配。

## 3．结论探讨

① **平均＋尾部约束在SPARCS仍保留有用收益，在L8当前网格只能回退Source。** SPARCS保留+7.76pp；L8放弃原+1.23或+0.94pp点估计，但这不是L8被独立证实的原增益。

② **改善尾部有明确代价。** SPARCS B3/B5分别放弃3.57/5.09pp目标mIoU，将T11降到3.29pp、最差ROI降到6.44pp；L8回退后本源支持损失为0，同时失去全部适配增量。尾均值是当前107 ROI的经验描述，不是总体风险上界；空间相关仍限制解释。

③ **值得另行检验训练阶段能否改善平均—尾部取舍。** 当前固定端点插值在L8不能同时保留非零适配和此尾部容忍度，SPARCS也仍有单ROI超5pp。结论限于这五α、5pp阈值和两个固定端点，不能推断所有训练方法均无法兼顾；本轮不增加α、门控、正则或数据池，也不启动训练。
'''
    assert len(report)<4200,len(report)
    (OUT/'RESULT.md').write_text(report,encoding='utf-8')
    (OUT/'README.md').write_text('# Phase215已完成\n\n先读RESULT.md。RULE_COMPARISON为六条预算线；ALL_ALPHA_CONSTRAINTS为两个头各五候选；SOURCE_SELECTION_LOCK在目标表连接之前封存。SOURCE_LEAVE_ONE_SELECTION及DETAILS、CHANGED_ROIS给出源留一敏感性，RULE_METRICS_SIDE_BY_SIDE保留全部代价。\n\n单位：IoU/mIoU为%，差和损失为pp；召回/FPR为0–1比例；空白为NA。放弃收益=旧−新，允许负值。SOURCE_SELECTION_LOCK中T11是带符号损失最差11项均值，不是所有ROI上界。\n\nL8回退Source；SPARCS三预算同选.25。探索回放不构成独立确认，三条预算线不是三个独立样本。原213/214封存只读。\n',encoding='utf-8')
    first=js(OUT/'SOURCE_STAGE_COST.json');second=js(OUT/'TARGET_JOIN_COST.json');elapsed=first['elapsed_seconds']+second['elapsed_seconds']+acceptance['elapsed_seconds'];memory=max(first['native_peak_working_set_bytes'],second['native_peak_working_set_bytes'],acceptance['native_peak_working_set_bytes'])
    assert elapsed<1800 and memory<4*1024**3
    scripts=[ROOT/'src/tools'/n for n in ['select_phase215_source.py','join_phase215_cached_targets.py','validate_phase215_independent.py','finalize_phase215_local.py']]
    save('CODE_HASHES.json',{'files':{str(p.relative_to(ROOT)):sha(p) for p in scripts},'protocol_sha256':sha(ROOT/'src/research_plans/PHASE215_SOURCE_MEAN_TAIL_SELECTION_20261023.md')})
    persist=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file());assert persist+1024**2<200*1024**2
    save('COST_RECEIPT.json',{'source_selection_and_LOO_seconds':first['elapsed_seconds'],'target_join_seconds':second['elapsed_seconds'],'independent_validation_seconds':acceptance['elapsed_seconds'],'recorded_compute_wall_seconds':elapsed,'native_peak_working_set_bytes':memory,'peak_kind':'Windows OS PeakWorkingSetSize for all three compute stages','persistent_snapshot_bytes':persist,'finalization_and_summary_reserved_bytes':1024**2,'combined_persistent_upper_bytes':persist+1024**2,'runtime_cap_seconds':1800,'memory_cap_bytes':4*1024**3,'persistent_cap_bytes':200*1024**2,'new_model_calls':0,'new_fits':0,'new_bootstrap_draws':0,'remote_connections':0,'downloads':0,'new_image_label_or_prediction_pixels_read':0,'failed_compute_attempts':0,'Phase213_214_all_original_SHA_verified_after_analysis':True})
    SUMMARY.mkdir(exist_ok=False)
    for n in ['RESULT.md','README.md','SOURCE_SELECTION_LOCK.json','RULE_COMPARISON.csv','ALL_ALPHA_CONSTRAINTS.csv','SOURCE_LEAVE_ONE_SELECTION.csv','SOURCE_SELECTION_CHANGED_ROIS.csv','INDEPENDENT_CPU_ACCEPTANCE.json','COST_RECEIPT.json','CODE_HASHES.json','INPUT_REGISTRY.json']:
        shutil.copyfile(OUT/n,SUMMARY/n)
    manifest={str(p.relative_to(OUT)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ['DELIVERY_MANIFEST.json','DELIVERY_COMPLETE.json']}
    save('DELIVERY_MANIFEST.json',manifest)
    save('DELIVERY_COMPLETE.json',{'phase':'215','status':'complete','manifest_sha256':sha(OUT/'DELIVERY_MANIFEST.json'),'artifacts':len(manifest),'source_selection_sealed_before_target_join':True,'independent_CPU_validation':'passed','Phase213_214_preserved':True,'model_calls':0,'remote_connections':0,'fits':0,'L8_new_alphas':[0,0,0],'SPARCS_new_alphas':[.25,.25,.25],'new_phase_or_training_started':False})
    shutil.copyfile(OUT/'DELIVERY_COMPLETE.json',SUMMARY/'DELIVERY_COMPLETE.json')
    assert all(sha(OUT/n)==r['sha256'] for n,r in manifest.items())
    total=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())+sum(p.stat().st_size for p in SUMMARY.rglob('*') if p.is_file());assert total<persist+1024**2
    print(json.dumps({'status':'complete','recorded_compute_seconds':elapsed,'native_peak_MiB':memory/1024**2,'combined_persistent_bytes':total,'report_characters':len(report),'artifacts':len(manifest)}))
if __name__=='__main__':main()
