"""Verify transferred receipts and publish the frozen-baseline branch experiment."""
from pathlib import Path
import json,csv,hashlib
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase420/phase420_20261101'
def js(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    for n,h in js(R/'TRANSFER_MANIFEST.json').items():assert sha(R/n)==h,n
    for n,h in js(R/'REPORT_COMPLETE.json')['files'].items():assert sha(R/n)==h,n
    for n,h in js(R/'REPORT_CODE_LOCK.json').items():assert sha(R/n)==h,n
    a=js(R/'INDEPENDENT_ACCEPTANCE.json');done=js(R/'COMPLETE.json');assert a['passed'] and done['passed']
    assert a['new_calls']==done['new_calls']==2192 and a['new_updates']==done['new_updates']==2000
    rows=list(csv.DictReader((R/'TARGET_SUMMARY.csv').open()));assert len(rows)==6
    for row in rows:
        cm=json.loads(row['confusion']);ii=[100*cm[i][i]/(sum(cm[i])+sum(r[i] for r in cm)-cm[i][i]) for i in range(3)]
        assert abs(sum(ii)/3-float(row['mIoU']))<1e-9
    end={x['mode']:x for x in rows if x['step']=='1000'};screen=js(R/'MECHANISM_SCREEN.json')
    assert abs(screen['margin_pp']-(float(end['spatial']['mIoU'])-max(float(end['global']['mIoU']),screen['baseline_mIoU'])))<1e-9
    assert a['forbidden_surface_cloud_switches']==0
    label=js(R/'LABEL_IDENTITY_AUDIT.json');assert label['passed'] and label['truth_arrays_exactly_equal_to_Phase411']==192 and label['label_edits']==0
    groups=list(csv.DictReader((R/'TARGET_GROUPS.csv').open()));assert len(groups)==192
    critical='S2A_MSIL1C_20180909T151911_N0206_R068_T25XEG_20180909T205122'
    analysis=['# Phase420 解释','','原有标签和完整32组均保留；本轮冻结的是已训练的Phase411目标基线，不是原始未适配Source。两组均在此基础上追加1000步。','',f"基线 {screen['baseline_mIoU']:.4f}%；整图校准 {float(end['global']['mIoU']):.4f}%；空间阴影修正 {float(end['spatial']['mIoU']):.4f}%。",'',f"空间分支超过较强基线/对照的余量 {screen['margin_pp']:+.4f} 个百分点；预设0.30门槛通过：{screen['clears_0_30pp']}。",'','原图逐像素检查：两个分支全部评价中，原基线地表直接变为云、或云直接变为地表的数量为0。阴影误报仍可能出现，必须看完整mIoU而不能只看该约束。','','T25XEG关键场景的第1000步混淆矩阵（行真值、列预测，地表/云/影）：']
    for mode in ['global','spatial']:
        g=next(x for x in groups if x['mode']==mode and x['step']=='1000' and x['product']==critical)
        analysis.append(f"- {mode}: {g['confusion']}；修复 {g['fixes']}，新增错误 {g['regressions']}。")
    analysis+=['','历史首次预检因JSON将裁剪尺寸tuple变为list而失败，发生于数据集构建、训练及前向之前。远端保留phase420_20261101_attempt0_config，零训练/零前向；模型加载和启动成本未独立计时。COST.csv为正式两组运行时间，COMPLETE.json含成功进程总体耗时，不把预检启动开销伪记为零。','','整图校准第1000步宏AP约68.5388，与基线相同；空间分支宏AP约68.5830，仅高0.0442点。整体mIoU增益很大程度已能由整图阴影校准实现，尚无强证据证明空间分支显著改善排序能力。空间分支修复119341像素、增加148412错误，总像素正确率下降但影IoU及三类等权mIoU上升；不得将mIoU提升改写成所有精度指标都改善。\n\n这是同参数量的整图/空间对照。两组损失均为每图出现类别等权像素NLL，与旧MsRE训练损失不同，不能把相对旧基线的差值全部归因于空间结构。只有空间与整图差值更直接对应空间输出的作用。局部亮度差/梯度尚未做单独消融，不能声称其云影专用性或论文新颖性已被验证。单seed development筛选，不作稳定性结论。']
    (R/'INTERPRETATION.md').write_text('\n'.join(analysis)+'\n',encoding='utf-8')
    (R/'README.md').write_bytes((R/'RESULT.md').read_bytes())
    title='## Phase420：冻结目标基线的阴影概率修正（2026-11-01）'
    entry=title+'\n\n- 固定Phase411第1000步基线，训练6817参数阴影分支；整图/空间两组各追加1000步。153 fit/32 development标签不变；seed40901。\n'
    for mode,v in end.items():entry+=f"- {mode}: mIoU {float(v['mIoU']):.4f}、影IoU {float(v['Shadow_IoU']):.4f}、影宏AP {float(v['macro_AP']):.4f}。\n"
    entry+=f"- 空间方案余量{screen['margin_pp']:+.4f}点，通过0.30门槛：{screen['clears_0_30pp']}。所有评价地表↔云直接切换为0，基线预测复现及冻结验收通过。\n- 2192次前向、2000次更新；预检配置类型失败零前向并归档。单seed结果，不自动延长训练。\n- outputs/phase420/phase420_20261101/RESULT.md。\n\n"
    log=ROOT/'src/EXPERIMENT_LOG.md';raw=log.read_bytes();marker='## 后续维护规则'.encode();assert raw.count(marker)==1
    if title.encode() not in raw:log.write_bytes(raw.replace(marker,entry.replace('\n','\r\n').encode()+marker))
    p=ROOT/'src/research_plans/phase420.md';s=p.read_text(encoding='utf-8');status='完成：2192次前向、2000次更新；原图比例约束、预测复现、冻结与指标独立验收通过，结果见outputs/phase420/phase420_20261101/RESULT.md。'
    if status not in s:p.write_text(s+'\n'+status+'\n',encoding='utf-8')
    (R/'DELIVERY_COMPLETE.json').write_text(json.dumps({'passed':True,'new_calls':2192,'new_updates':2000,'local_mIoU_recomputed':True,'forbidden_surface_cloud_switches':0,'screen_passed':screen['clears_0_30pp']},indent=2))
    manifest={p.relative_to(R).as_posix():sha(p) for p in R.rglob('*') if p.is_file() and p.name!='LOCAL_MANIFEST.json'}
    (R/'LOCAL_MANIFEST.json').write_text(json.dumps(manifest,indent=2));print('sealed',len(manifest),'files')
if __name__=='__main__':main()
