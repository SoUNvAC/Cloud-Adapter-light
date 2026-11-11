"""Separated data/interpretation report and final immutable manifest."""
from pathlib import Path
import csv,json,hashlib,shutil,statistics,time,subprocess,sys
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/phase216/phase216_20261023';RAW=OUT/'delivery';SUMMARY=ROOT/'src/research_results/phase216_20261023'
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rd(n):return list(csv.DictReader((OUT/n).open(encoding='utf-8-sig')))
def main():
    assert js(OUT/'INDEPENDENT_CPU_ACCEPTANCE.json')['status']=='passed'
    subprocess.run([sys.executable,'-X','utf8','-B',str(ROOT/'src/tools/plot_phase216.py')],check=True)
    rows=rd('ALL_CANDIDATES.csv');joint=rd('JOINT_RULE.csv');ci=rd('PAIRED_CI.csv');dom=rd('FINITE_SET_DOMINANCE.csv');cost=js(RAW/'COMPLETE.json');remote=js(RAW/'REMOTE_CPU_ACCEPTANCE.json')
    table='\n'.join(f"| {r['arm']} | {float(r['alpha']):g} | {float(r['pool_loss_pp']):.3f} | {float(r['T11_pp']):.3f} | {float(r['worst_ROI_loss_pp']):.3f} | {r['ROI_loss_gt5']} | {float(r['val_mIoU']):.3f} | {float(r['test_mIoU']):.3f} | {float(r['test_Shadow_IoU']):.3f} | {100*float(r['test_Shadow_recall']):.2f} | {100*float(r['test_FPR']):.3f} |" for r in rows)
    jtable=[]
    for r in joint:
        b=int(r['budget_pp']);c=next(x for x in ci if int(x['budget_pp'])==b and x['metric']=='test_mIoU_D_minus_C_pp')
        jtable.append(f"| {b} | {r['C_alpha']} / {r['D_alpha']} | {float(r['C_source_pool_loss_pp']):.3f} / {float(r['D_source_pool_loss_pp']):.3f} | {float(r['C_T11_pp']):.3f} / {float(r['D_T11_pp']):.3f} | {float(r['C_test_mIoU']):.3f} / {float(r['D_test_mIoU']):.3f} | {float(r['D_minus_C_test_mIoU_pp']):+.3f} [{float(c['CI95_low']):+.3f}, {float(c['CI95_high']):+.3f}] |")
    cend=next(r for r in rows if r['arm']=='C' and float(r['alpha'])==1);dend=next(r for r in rows if r['arm']=='D' and float(r['alpha'])==1)
    dominated=sorted({float(r['C_alpha']) for r in dom if r['D_dominates_C']=='True'})
    tc=js(RAW/'C_TRAIN_COMPLETE.json');td=js(RAW/'D_TRAIN_COMPLETE.json')
    lossstats={}
    for arm in ['C','D']:
        ls=[json.loads(x) for x in (RAW/f'{arm}_LOSSES.jsonl').read_text().splitlines()]
        lossstats[arm]={k:{'first100':statistics.mean(r[k] for r in ls[:100]),'last100':statistics.mean(r[k] for r in ls[-100:]),'min':min(r[k] for r in ls),'max':max(r[k] for r in ls)} for k in ['target_loss','source_KL']}
    report=f'''# Phase216：SPARCS头部适配＋源预测保持

## 1．数据结果

C普通目标监督、D同监督＋固定λ=1/T=1源语义KL，各seed21601、4000步、1,316,516个相同head参数，只用末步；同Source、同4000步裁剪/翻转/光度序列。SPARCS原60训练景使用全部密集标签，val10/test10分开评价。Source此前为监督模型（官方train池8490，已知parent4000步，继承初始化的早期训练成本未重建）。新增256张源train RGB仅D使用，每张15或16次，另1次门检；不读源train/test标签。哈希规则按官方index字符串取前256，PNG逐张匹配旧官方原数组审计；与535val/975test无ID或内容重复。与test有1个相同产品、无相同ROI，完整地理独立性未宣称。

源评价535图/107 ROI；T11为107 ROI中最差11个带符号mIoU损失的均值，损失为Source−候选，负值不截断。源值及目标mIoU/IoU单位%，损失单位pp；召回/FPR栏为%。Source公共端点重复列出，不是两份独立证据。

| 臂 | α | 源池化损失 | T11 | 最差ROI | >5pp ROI数 | val mIoU | test mIoU | test影IoU | 影召回% | FPR% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{table}

先封存仅源选择锁，再连接目标指标；B=1/3/5且T11≤5，取最大可行α，0回退。未按test选α。配对bootstrap10000、seed216023，10个原test scene；CI条件于固定模型和选定α，不涵盖训练、回放池及选择不确定性。

| B | C/D α | C/D源池化损失 | C/D T11 | C/D test mIoU | D−C test差及95%CI |
|---|---|---|---|---|---|
{chr(10).join(jtable)}

`PAIRED_CI.csv`另报按107 ROI配对重采样的源池化损失差、每次重排的T11差；源数据已用于选择，区间不作总体保证。`ALL_POOL_METRICS/PER_GROUP/ERROR_FLOWS.csv`完整保留三类IoU、逐组均值/召回/FPR/最大误报/零召回/六方向净错误流。`FINITE_SET_DOMINANCE.csv`完整25对，D在test mIoU不低、T11不高且一项严格改善的二维点估计支配覆盖C的α={dominated}；并列源池化及最差ROI代价均保留。

实际{cost['total_units']:,}单位调用：{cost['counts']}；完整segmentor内部backbone/head不重复记，共享源backbone/教师head/学生head分列。含首次评价失败535次及两次数值诊断，均未重训。原因是训练结束遗留CuDNN确定性设置；恢复原评价默认设置后，Source源/val/test三份混淆均逐项复现。GPU阶段{cost['GPU_stage_seconds']:.1f}s；C/D训练{tc['train_seconds']:.1f}/{td['train_seconds']:.1f}s，D−C额外{td['train_seconds']-tc['train_seconds']:.1f}s。GPU allocated/reserved已观测累计峰值{cost['GPU_peak_allocated_bytes']/1024**3:.2f}/{cost['GPU_peak_reserved_bytes']/1024**3:.2f}GiB，CPU峰值{cost['CPU_peak_bytes']/1024**3:.2f}GiB；旧失败进程最终瞬间峰值未单独封存，其值取15秒守卫的累计峰值采样。资源/物理盘采样、失败成本和SHA完整保留。两头同初始态，非head/teacher不变；门检loss一致、初始KL近0、梯度白名单通过，所有内部α逐张量CPU float64重算通过。原训练warn_only保留，不把固定seed称完全位级确定性。

## 2．结论探讨

源预测保持在本轮改善了已测有限集合的取舍。α=1时，目标test mIoU仅差+0.093pp（71.696→71.788），源池化损失却减少3.501pp，T11减少8.627pp；α=0.5/0.75/1的D各自二维点估计支配同α的C。α=0.25未被D覆盖，不能说整条曲线处处更好；D完整端点的T11仍为11.702pp，也未满足5pp尾部约束。

这不表现为简单压低适配强度：D末步目标性能未下降，参数相对Source的L2改变量仅从4.8708降至4.8259（约−0.92%），而源损伤下降较多。它支持保留蒸馏作为基线，尚不能证明机制或排除未测试插值点的其他取舍。

原联合规则三个预算均选择C=0.25、D=0.5，是同一对候选的重复列示，非三次独立验证。D的test收益+3.562pp，条件95%CI[1.861,5.694]；代价是源池化损失多0.405pp，CI[0.011,0.752]。T11多0.148pp，CI[−2.779,2.418]，不能声称所选D比所选C的源尾部更低；其最差ROI损失8.561pp也说明平均尾部限制不保证每ROI安全。

新增256张源RGB、重复4000次以及约52.1%训练时间增量，在这一固定配方下换来了有价值的目标—保源取舍，值得作为已有保源对照保留。单seed、固定λ/回放池，且SPARCS test此前已查看：这是探索结果，不能泛化为解决遗忘或独立确认。损失曲线/原始值完整保留，不追加seed、λ、回放池或新机制。
'''
    (OUT/'RESULT.md').write_text(report,encoding='utf-8')
    (OUT/'LOSS_SUMMARY.json').write_text(json.dumps(lossstats,indent=2),encoding='utf-8')
    from analyze_phase216 import peak
    memory=peak();assert memory<16*1024**3
    (OUT/'LOCAL_REPORT_COST.json').write_text(json.dumps({'native_peak_working_set_bytes':memory,'disk_free_bytes':shutil.disk_usage(ROOT).free,'new_model_calls':0,'plotting_backend':'existing ReportLab and Poppler','environment_changes':0},indent=2),encoding='utf-8')
    # Source data/heads remain remote. Only scoped small reports are delivered.
    SUMMARY.mkdir(parents=True,exist_ok=False)
    for n in ['RESULT.md','ALL_CANDIDATES.csv','JOINT_RULE.csv','PAIRED_CI.csv','FINITE_SET_DOMINANCE.csv','INDEPENDENT_CPU_ACCEPTANCE.json','LOSS_SUMMARY.json']:shutil.copy2(OUT/n,SUMMARY/n)
    print(json.dumps({'report':str(OUT/'RESULT.md'),'dominates_C_alphas':dominated}))
if __name__=='__main__':main()
