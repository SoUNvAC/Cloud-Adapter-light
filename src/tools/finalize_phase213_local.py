"""Render complete Phase213 results and sealed delivery manifest after acceptance."""
from pathlib import Path
import argparse,csv,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/phase213/phase213_20261021';RAW=OUT/'delivery'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def mdtable(rows,columns):
    def fmt(v):
        if v is None:return 'NA'
        if isinstance(v,float):return f'{v:.4f}'
        return str(v)
    return '\n'.join(['| '+' | '.join(title for key,title in columns)+' |','|'+'|'.join('---' for _ in columns)+'|']+['| '+' | '.join(fmt(r.get(k)) for k,title in columns)+' |' for r in rows])
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--state',type=int,choices=[1,2,3],required=True);args=parser.parse_args()
    s=read(OUT/'ANALYSIS_SUMMARY.json');assert read(OUT/'ANALYSIS_ACCEPTANCE.json')['status']=='passed';assert read(RAW/'REMOTE_CPU_ACCEPTANCE.json')['status']=='passed'
    c=read(RAW/'RUN_COMPLETE.json');adm=read(RAW/'FINAL_ADMISSION_LOCK.json');groups=list(csv.DictReader((OUT/'PER_GROUP_CHANGES.csv').open(encoding='utf-8-sig')))
    budget=mdtable(s['budgets'],[('target','目标'),('budget_pp','源预算pp'),('alpha','锁定α'),('source_pooled_mIoU','源val mIoU'),('source_loss_pp','源损失pp'),('test_pooled_mIoU','test mIoU'),('test_gain_vs_Source_pp','test增益pp'),('CI95_low','增益CI下界'),('CI95_high','增益CI上界'),('test_Shadow_IoU','Shadow IoU')])
    candidate=mdtable(s['candidates'],[('target','目标'),('alpha','α'),('source_pooled_mIoU','源mIoU'),('test_pooled_mIoU','test池化mIoU'),('test_group_mean_mIoU','逐scene平均mIoU'),('test_Surface_IoU','Surface IoU'),('test_Cloud_IoU','Cloud IoU'),('test_Shadow_IoU','Shadow IoU'),('test_Shadow_recall','Shadow召回'),('test_FPR','非影→影FPR'),('test_max_group_FPR','最大scene FPR'),('test_zero_recall_groups','零召回scene')])
    history=mdtable(s['history'],[('target','目标'),('method','历史方法'),('source_pooled_mIoU','历史源val mIoU'),('historical_target_val_mIoU','历史target-val mIoU')])
    worst=[]
    for r in s['budgets']:
        for pool in ['source','test']:
            selected=[x for x in groups if x['target']==r['target'] and x['pool']==pool and float(x['alpha'])==r['alpha']]
            x=min(selected,key=lambda x:float(x['delta_mIoU_pp']));worst.append(dict(target=r['target'],budget=r['budget_pp'],pool=pool,group=x['group'],delta_mIoU_pp=float(x['delta_mIoU_pp'])))
    worsttable=mdtable(worst,[('target','目标'),('budget','预算pp'),('pool','池'),('group','最差ROI/scene'),('delta_mIoU_pp','相对Source变化pp')])
    states={1:'源预算规则在两个target-test均保留有用目标收益',2:'只在一个target-test成立',3:'target-test不支持稳定收益'}
    data=f'''# Phase213：源预算插值规则的目标测试\n\n## 1．数据结果\n\n使用Phase212-R锁定端点及六个已有插值头，零训练、零优化器更新。Source在CloudSEN12 High val535张／107 ROI重新前向，预测SHA、全部ROI混淆与Phase212-R FP32逐项一致，池化mIoU 77.8152217929%。随后冻结全部候选头SHA、源val缓存、预算选择、映射与输入manifest，再首次读取本轮目标test像元。\n\nL8原target-test为2167张／16 scenes，SPARCS为10张／10 scenes。L8按本协议原manifest完整test，不再用历史val的Shadows=yes子集筛选。均RGB512／三父类／FP32；CloudSEN官方test未读，target-train/val未用于本轮选择。\n\nIoU/mIoU单位%；差值pp；召回和FPR为0–1比例。池化mIoU来自相加混淆；逐scene均值另报。\n\n### 三条预注册预算线\n\n{budget}\n\nL8 α=.5源损失3.03459065599pp，超过B3，故B1/B3均选.25；没有四舍五入后纳入B3。各行CI是相对同池Source的池化mIoU差，10000次配对整ROI/scene bootstrap，seed213020。源val用于选择，不能称独立确认；测试CI条件于这一个Source和重训端点，不含训练/选模不确定性。\n\n### 全部固定候选\n\n{candidate}\n\n逐组指标、缺类、无影NA、零召回、ignore、完整混淆及错误流见PER_GROUP.csv、PER_GROUP_CHANGES.csv、ERROR_FLOWS.csv与delivery/evaluation原始JSON。源107 ROI不是535独立场景；目标16/10 scenes也不因像元数扩大样本量。所有候选和预算完整保留，没有按test选α。\n\n### 最差局部损伤（每条预算线）\n\n{worsttable}\n\n### 历史描述性参照，不能与test作严格胜负\n\n{history}\n\n上述历史目标成绩均为target-val／FP16；shared-parent仅最终审计舍入值，LoRA目标值仅两位小数。当前为不同支持的target-test／FP32，不能宣称插值在同一测试支持上超过MsRE/LoRA，不能用test数值与旧val坐标画严格支配关系。旧共享头已由重训新头替代，α=1不是旧SHA权重。\n\n### 正确性、资源和失败记录\n\n实际新增前向{c['calls']}：Source535＋目标10885；零训练。运行阶段墙钟{c['elapsed_seconds']:.2f}秒，非纯GPU kernel时间；GPU分配峰值{c['GPU_peak_allocated_bytes']/1024**3:.3f}GiB、进程RSS峰值{c['RSS_peak_bytes']/1024**3:.3f}GiB。宿主物理卷启动余量见HOST_DISK.json，持久文件和包以最终COST_RECEIPT.json为准。\n\n前三个目录保留：锁路径失败、包校验部署失败、非持久buffer验收失败；各0次前向。run03额外状态诊断也0次前向，发现仅data_preprocessor.mean/std两个非持久buffer，381个持久张量全等；run04显式核对这两个固定值及完整冻结状态。未删除任何旧结果或重置账本。失败GPU构建时间以保留诊断回执为准，未把其算成训练。\n\n独立CPU验收：六个头用另一种加权和公式核对，形状/dtype/SHA一致；11420条调用逐条连续且各池计数准确；各图→组→池混淆精确相等；独立标量IoU及直接抽样重算、同参考支持与未舍入最大可行α均通过。运行时非head参数/buffer每次加载及评价后全等。\n'''
    quality=read(OUT/'TEST_LABEL_METADATA_AUDIT.json')
    data+='\n### 数据边界与缺项\n\nPhase212-R未评价target-test，因此不存在可供本轮Source test逐像元复现的旧test基准。只对同支持源val做预测SHA/混淆精确复现；两个test的Source均为本次新测α=0，并与本轮其他α严格同支持比较，没有拿旧val成绩冒充test复现。历史MsRE/LoRA也缺同支持test参照，此项严格比较明确未完成，不能据不同池的数值宣布超过旧方法。\n\n### L8标签完整性限制\n\n仅用原USGS元数据复核，16个L8 test scene中6个Shadows=yes、10个Shadows=no；后10组在所给掩膜中也无Shadow参考像元。Shadows=no不能证明画面内真实没有云影。这些组仍按预先封存的完整manifest参与原始指标，但其漏标可能影响Shadow IoU、三类mIoU和名义FPR；不能将其当作已人工确认的纯负例，也不能把这些FPR当作可靠真实误报率。未在看过结果后筛选或重算一个较好看的主评价池。具体清单见TEST_LABEL_METADATA_AUDIT.json。\n'
    analysis='''## 2．结论分析\n\n**SPARCS保留了有用的目标收益，L8尚未确认稳定收益，因此本轮概括为“只在一个target-test成立”。** 这只针对固定checkpoint和原协议，不是跨seed或跨传感器的普遍结论。\n\n1. **SPARCS：预算规则有效保留收益。** B1/B3/B5相对Source的test池化mIoU增益分别为+7.7582/+11.3265/+12.8458pp，三个95%区间均高于0；相应源损失仅0.3806/1.8092/4.2893pp。三条预算线上10/10个scene的mIoU点估计均提高，仍保留每个scene的原值和区间。α=.75的test mIoU71.2649%，接近端点α=1的71.4107%，同时少损失源性能3.6703pp。该点由B5源预算预先选出，不能将此事后近似性称作额外选模证据。\n\n2. **L8：正点估计不等于确认通过。** B1/B3同选α=.25，test增益+1.2326pp，95%CI[−0.0978,2.5776]；B5选α=.5，+0.9359pp，CI[−2.2334,4.0301]，均跨零。增加α时源能力持续下降，目标mIoU却先升后降，端点α=1甚至较Source下降2.8952pp。预算规则避免直接选中该端点，但没有证明稳定正收益。L8还存在10个Shadows=no场景的参考完整性限制，不能把完整清单的云影指标解释为可靠精细真值下的性能确认。\n\n3. **池化保源不等于每个ROI保源。** L8 B1的源池化损失仅0.4115pp，但最差ROI_0626下降13.0356pp；B5最差ROI_0942下降27.5691pp。SPARCS B5最差ROI_0626也下降21.2789pp。必须保留这些损伤，不能宣称无遗忘或所有场景均满足预算。\n\n4. **实际决策有不同代价。** SPARCS B1的池化FPR由Source3.2432%略增至3.3225%；B3/B5为3.1815%/2.9130%，最大scene FPR分别10.7060%/7.3117%/4.8595%。本轮没有1%误报门槛，也未扫描阈值，因此不能宣称满足其他phase的部署约束。L8名义FPR另受不完整云影参考影响。\n\n三条预算线是同一规则的预设工作点，不能事后只保留最好的一条。全候选test曲线仅用于解释取舍，不用于修改候选。源val参与选择，区间条件于一个Source和一个重训端点，不包含训练/选模不确定性；10个SPARCS scene仍是小样本，像元数不扩充独立样本量。VFM预训练完整覆盖未知，target-test只指原协议内未参与本轮模型/α选择。\n\n头部插值为既有WiSE-FT类基线，不是新算法，也不证明通用DINO能力保持。历史MsRE/LoRA只有不同支持的target-val／FP16成绩，不能据当前test数值判严格胜负；关于超过历史MsRE的匹配test比较仍缺证据。没有追加α、训练、seed、阈值搜索或下一phase。\n'''
    (OUT/'DATA_RESULTS.md').write_text(data,encoding='utf-8');(OUT/'CONCLUSIONS.md').write_text(analysis,encoding='utf-8')
    (OUT/'RESULT.md').write_text(data+'\n'+analysis,encoding='utf-8')
    (OUT/'README.md').write_text('# Phase213交付\n\n结果和解释分别见DATA_RESULTS.md、CONCLUSIONS.md，完整合并版RESULT.md。预算表BUDGET_TEST_TABLE.csv，全部候选及不同支持的历史参照PARETO_TEST_TABLE.csv；逐组、错误流、bootstrap、原始混淆与验收齐备。\n\n状态仅概括本轮固定工作点；数值、区间和局部损伤限制以RESULT.md为准，不是新增统计通过规则。\n\n'+states[args.state]+'\n',encoding='utf-8')
    summary=ROOT/'src/research_results/phase213_20261021';summary.mkdir(exist_ok=True)
    for n in ['RESULT.md','README.md','BUDGET_TEST_TABLE.csv','PARETO_TEST_TABLE.csv','PAIRED_DIFFERENCES_CI.csv','ANALYSIS_ACCEPTANCE.json']:
        shutil.copyfile(OUT/n,summary/n)
    print(states[args.state])
if __name__=='__main__':main()
