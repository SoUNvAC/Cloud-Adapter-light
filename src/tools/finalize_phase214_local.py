"""Write the short Phase214 report, resource receipt and final SHA seal."""
from pathlib import Path
import csv,hashlib,json,shutil,time
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/phase214/phase214_20261023';OLD=ROOT/'outputs/phase213/phase213_20261021';SUMMARY=ROOT/'src/research_results/phase214_20261023'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(n):return list(csv.DictReader((OUT/n).open(encoding='utf-8-sig')))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def main():
    accepted=read(OUT/'INDEPENDENT_CPU_ACCEPTANCE.json');assert accepted['status']=='passed'
    ci=rows('L8_STRATIFIED_DIFFERENCES_CI.csv');strata=rows('L8_STRATIFIED_RESULTS.csv');tails=rows('SOURCE_TAIL_SUMMARY.csv');selected=[r for r in tails if r['budget_labels']]
    layerlines=[]
    for a in [.25,.5,.75]:
        full=next(r for r in strata if r['support']=='full' and float(r['alpha'])==a);yes=next(r for r in ci if r['support']=='yes' and float(r['alpha'])==a);no=next(r for r in ci if r['support']=='no' and float(r['alpha'])==a)
        layerlines.append(f"| {a:g} | {float(full['delta_mIoU_pp']):+.2f} | {float(yes['delta_mIoU_pp']):+.2f} [{float(yes['CI95_low']):.2f}, {float(yes['CI95_high']):.2f}] | {float(no['delta_mIoU_pp']):+.2f} [{float(no['CI95_low']):.2f}, {float(no['CI95_high']):.2f}] |")
    taillines=[f"| {r['target']} | {r['budget_labels']} / {float(r['alpha']):g} | {float(r['source_pooled_loss_pp']):.2f} | {float(r['worst11_mean_loss_pp']):.2f} | {float(r['worst_ROI_loss_pp']):.2f} ({r['worst_ROI']}) | {r['ROIs_loss_gt_5pp']} |" for r in selected]
    report='''# Phase214：L8参考支持分层与源ROI尾部审计

## 1．数据结果

仅用213已封存混淆及元数据，本地CPU，零训练／拟合／前向／远程连接／像元读取。SHA与全部五α、20组模型/池指标及原预算选择已复现。Source均为213 FP32 α=0；213结果和选择未改写。

**L8分层。** 完整16景，yes六景，no十景。下表为同层池化ΔmIoU（pp），方括号为分层配对scene bootstrap 95%CI，10000次、seed214023；完整池只复现213点值。

| α | 完整16景Δ | yes六景Δ [CI] | no十景Δ [CI] |
|---|---:|---:|---:|
'''+ '\n'.join(layerlines)+'''

五α的池化/逐景均值、三类IoU、召回、名义FPR及支持数在L8_STRATIFIED_RESULTS.csv。no层无参考Shadow，召回均NA；不是0，也不证明真实无影。yes也未经本轮人工认证。α=.5留一scene的Δ范围：完整[0.49,1.98]、yes[2.11,3.38]、no[−3.41,−0.81]；这是单景敏感性范围，不是CI。

α=.5时，yes层总误分净减4,514,693（有效像元的−2.3055pp），no层净增1,221,031（+0.3280pp），相加精确复现完整池净减3,293,662。no层Cloud→Surface净增15,842,734，不能把层间差异全归因为云影漏标。六方向完整净变化见L8_ERROR_NET_CHANGES.csv；mIoU差不可相加算“贡献百分比”。

**源107 ROI。** 正值为相对Source的损失，单位pp。尾均值固定为最差11/107 ROI算术平均。B1/B3在L8是同一工作点，不是两次复现。

| 目标 | 预算 / α | 池化损失 | 最差11均损失 | 最差ROI损失 | 损失>5 ROI数 |
|---|---|---:|---:|---|---:|
'''+ '\n'.join(taillines)+'''

完整五α的均值、中位数、P90、>1/3/5/10计数见SOURCE_TAIL_SUMMARY.csv。所有源ROI各候选有效IoU类数均为3，未出现平均分母类数跳变。ROI_0942在L8 α=.5的Shadow IoU从62.57%降至6.39%，错分率增加8.84pp；ROI_0626在L8 α=.25错分率增加10.93pp。这些损伤不只是有效类数变化造成的算术现象。

L8 α=.25各ROI**正的误分像元净增量**合计820,365，错误净减少另计587,836；最大1/5/10 ROI占正增量17.46%/38.64%/57.74%。这不是mIoU贡献比例，也不是具体像元新增/消除错误的转移统计。所有点见SOURCE_ERROR_CONCENTRATION.csv。

同预算两头损失>5pp的ROI交/并集：B1为1/3，交集ROI_0626；B3为2/15，交集ROI_0124、ROI_0626；B5为13/49。完整ID、最差10名单、参考类构成和六方向净流均保留，不从ID猜地物。L8 α=.25源逐ROI留一Δ范围[−0.4682,−0.2713]；α=.5为[−3.1611,−2.8476]。源区间仅引用213原107 ROI bootstrap，没有另抽更有利区间。

## 2．结论探讨

① **L8呈现标签支持依赖。** 完整池仍是213的弱且不确定收益；yes层在α=.25/.5有更清楚的正向描述，no层随α增加更不利。六景事后分层不能替换213主终点或宣布独立确认成功。混淆不足以估计漏标率，差异不证明“L8失败由漏标导致”。

② **池化预算掩盖明显局部代价。** L8 B1池化仅损失0.41pp，却有31 ROI损失>1pp、3 ROI>5pp，最差13.04pp；B5有24 ROI>5pp、8 ROI>10pp。局部损伤并非有效类数跳变，池化留一相对稳定也不表示每ROI安全。107 ROI可能相关，尾均值不是总体风险上界或每景保证。

③ **优先补清L8参考支持的解释，再研究保源选择规则。** 目前更值得把可靠评价边界和平均/尾部代价说清楚，暂不足以因yes子集较好就继续叠加机制。未来若研究尾部约束，应另立协议冻结规则；本轮不据尾部排名重选α、改预算、设计门控或训练。

## 3．交付与验收

独立脚本未导入主脚本函数，另一逐类公式复核1200个原逐组指标、分层计数、全部尾部名单、净错误集中度、802个留一结果及六套10000次抽样与手工分位数。零正类、缺类和零分母按NA处理。一次验证器迭代器错误、一次新增检查变量覆盖错误均留档；修复后14个既有CSV SHA完全一致，设置未变。成本、代码/输入SHA、验收与完整明细随附；未改213封存、push或启动后续实验。
'''
    assert len(report)<4600,(len(report),'short report budget')
    (OUT/'RESULT.md').write_text(report,encoding='utf-8')
    (OUT/'DATA_DICTIONARY.md').write_text('''# 数据口径

IoU/mIoU、损失、差值、net_pct_valid均以百分数或百分点表示。recall/FPR/fraction/share/error_rate为0–1比例。CSV空白表示NA，不补0。confusion行为Surface/Cloud/Shadow参考，列为同顺序预测；union为0的类IoU不参与该组mIoU，参考Shadow为0时召回NA。

L8 full/yes/no分别16/6/10 scene，no不表示真实无影。L8_SCENE_RESULTS中同一景会在full和所属层各出现一次，不能将整张表再次相加。L8三层误分计数可按yes+no=full核对，mIoU差不能相加。留一敏感性不是置信区间。

源表loss=Source-current，正为损伤；预算只按源池化损失选择，原α不变。最差11固定ceil(107*10%)，相同损失按ROI ID字典序列出；α=0全为0，所列前10/11仅是并列的确定性顺序。参考类组成来自已交付混淆行和，不是新读标签或可部署风险特征。所有流仅为净计数变化，不标识具体像元转移。

集中度分母为各ROI正的总误分净增量之和；错误减少另列有符号负值及正幅度。分母0时share空白。阈值>1/3/5/10严格使用原浮点损失，P90/median及bootstrap分位数用线性插值。

BOOTSTRAP_RULE记录固定214023流，顺序yes/no，三内部α共享同层配对抽样。L8_BOOTSTRAP_DRAWS保留全部60000个差值；SOURCE_EXISTING_213_CI逐行引用原区间，未重抽源bootstrap。
''',encoding='utf-8')
    for name,h in read(OUT/'INPUT_HASHES.json')['files'].items():assert sha(OLD/name)==h
    original=read(OLD/'DELIVERY_MANIFEST.json');assert all(sha(OLD/n)==r['sha256'] and (OLD/n).stat().st_size==r['bytes'] for n,r in original.items())
    codefiles=[ROOT/'src/tools'/n for n in ['analyze_phase214_local.py','validate_phase214_independent.py','finalize_phase214_local.py']]
    save(OUT/'CODE_HASHES.json',{'files':{str(p.relative_to(ROOT)):sha(p) for p in codefiles},'protocol':sha(ROOT/'src/research_plans/PHASE214_L8_SUPPORT_SOURCE_TAIL_AUDIT_20261023.md')})
    a=read(OUT/'RUN_STATUS.json');initial=read(OUT/'INITIAL_ANALYSIS_COST.json');peak=max(a['native_peak_working_set_bytes'],initial['native_peak_working_set_bytes'],accepted['native_peak_working_set_bytes'])
    elapsed=initial['elapsed_seconds']+a['elapsed_seconds']+accepted['elapsed_seconds']+.25
    assert elapsed<1800 and peak<4*1024**3
    persist=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file());assert persist<200*1024**2
    cost={'recorded_analysis_and_validation_wall_seconds':elapsed,'analysis_first_run_seconds':initial['elapsed_seconds'],'analysis_verification_refresh_seconds':a['elapsed_seconds'],'final_independent_validation_seconds':accepted['elapsed_seconds'],'first_successful_validator_seconds':.25,'failed_validator_and_preflight_elapsed_seconds':None,'elapsed_unknown_reason':'two short local assertion failures were not timed inside their scripts; command diagnostics retained; do not invent precise duration','native_process_peak_working_set_bytes':peak,'peak_sampling':'Windows OS peak working-set counter for analysis and successful independent validator processes; failed short validator/preflight peaks not sampled','Python_tracemalloc_peak_bytes':max(a['Python_tracemalloc_peak_bytes'],initial['Python_tracemalloc_peak_bytes']),'persistent_snapshot_bytes':persist,'persistent_limit_bytes':200*1024**2,'memory_limit_bytes':4*1024**3,'runtime_limit_seconds':1800,'neural_calls':0,'fits':0,'remote_connections':0,'downloads':0,'image_label_score_pixels_read':0,'new_source_bootstrap_draws':0,'target_strata_draws_per_stratum':10000,'verification_refresh':'fixed same seed/config; 14 existing CSVs SHA unchanged','Phase213_all_original_artifacts_readonly_verified':len(original)}
    save(OUT/'COST_RECEIPT.json',cost)
    (OUT/'README.md').write_text('# Phase214完成\n\n先读RESULT.md（两页以内的简短结果/结论），DATA_DICTIONARY.md解释NA、单位、符号及重复支持。完整五α、三支持、源107 ROI与尾部均已交付。\n\n本地CPU，零推理/训练/远程连接；Phase213源预算和原结论未修改。分层是事后描述，不能将yes六景替换原主终点。\n',encoding='utf-8')
    SUMMARY.mkdir(exist_ok=False)
    for name in ['RESULT.md','README.md','L8_STRATIFIED_RESULTS.csv','L8_STRATIFIED_DIFFERENCES_CI.csv','SOURCE_TAIL_SUMMARY.csv','SOURCE_BUDGET_ROI_OVERLAP.csv','INDEPENDENT_CPU_ACCEPTANCE.json','COST_RECEIPT.json','INPUT_HASHES.json','CODE_HASHES.json']:
        shutil.copyfile(OUT/name,SUMMARY/name)
    manifest={str(p.relative_to(OUT)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ['DELIVERY_MANIFEST.json','DELIVERY_COMPLETE.json']}
    save(OUT/'DELIVERY_MANIFEST.json',manifest)
    save(OUT/'DELIVERY_COMPLETE.json',{'phase':'214','status':'complete','manifest_sha256':sha(OUT/'DELIVERY_MANIFEST.json'),'artifacts':len(manifest),'Phase213_preserved':True,'independent_validation':'passed','new_model_calls':0,'remote_connections':0,'new_phase_started':False,'conclusion':'L8 descriptive result depends on label-support stratum; pooled source budgets hide localized harm; prioritize data interpretation and source-tail selection rules, no rule changed here'})
    shutil.copyfile(OUT/'DELIVERY_COMPLETE.json',SUMMARY/'DELIVERY_COMPLETE.json')
    assert all(sha(OUT/n)==r['sha256'] for n,r in manifest.items())
    finalbytes=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())+sum(p.stat().st_size for p in SUMMARY.rglob('*') if p.is_file());assert finalbytes<200*1024**2
    print(json.dumps({'status':'complete','report_characters':len(report),'native_peak_MiB':peak/1024**2,'recorded_runtime_seconds':elapsed,'combined_new_bytes':finalbytes,'artifacts':len(manifest)},ensure_ascii=False))
if __name__=='__main__':main()
