"""Editorial clarification of actual tradeoffs; all raw numeric tables remain immutable."""
from phase310_common import *
def main():
    root=LOCAL/'remote_backup';b=load(LOCAL/'BACKUP_RECEIPT.json');assert b['all_stream_members_verified']
    raw=root/'RESULT.md';text=raw.read_text(encoding='utf-8');original_csv={p.name:sha(p) for p in root.glob('*.csv')}
    old='主比较Q90-SP−Q90-PL组均mIoU -6.339249pp。修复原D与原D外新增损伤必须同时看；保留纠正不能用错误净减少数代替。'
    new='本次固定lambda=1未取得更好的总体取舍：Q90-SP−Q90-PL组均mIoU −6.339249pp；Surface IoU提高，但Cloud/Shadow IoU及Shadow AP降低。原D修复371012（73.207%），同时原D外新增Surface→Cloud43277，另有错误类别转移。原Cloud纠正仅保留362132/784478（46.162%），417682退回Shadow、4664转Surface。相对Q90-PL，Cloud→Shadow错误增加613146，Surface→Shadow增加102797；Surface→Cloud虽净减少339644，不能单独宣称成功。'
    assert old in text;text=text.replace(old,new)
    old='最大历史损伤组T25XEG：'
    text=text.replace(old,'最大历史损伤组T25XEG不能仅用mIoU判断像元修复：原D206076中188696修复为Surface；Surface IoU由80.270066%升至95.168236%，但引入33087个Shadow假阳性，union类数由2变3，使组mIoU按原缺类规则降低。该组仍未恢复Source全Surface结果。原始指标：')
    needle='目标训练入口不读参考、场景标签或D/K包；'
    text=text.replace(needle,'两项loss和梯度确实生效：PL/SP平均值0.30225094/0.042682604；固定101/500/900步SP梯度范数为0.043107/0.464186/1.329087，初始零LoRA的SP=0。P只是全图Source-Surface未接纳区域，同512软目标仍为三类分布，不能把它说成硬Surface监督，也不由梯度非零证明因果。\n'+needle)
    (S/'RESULT.md').write_text(text,encoding='utf-8')
    assert all(sha(S/n)==h for n,h in original_csv.items())
    (S/'TABLE_README.md').write_text('''# 表口径

- ERROR_FLOW/BOUNDARY_ERROR_FLOW及TRANSITIONS27复用原304字段名：Source_errors、Source_pred表示comparison箭头左端旧模型；student_errors、student_pred表示箭头右端模型。Q90-PL->Q90-SP行的Source字段实际指Q90-PL，不是Source-only。
- DAMAGE_BENEFIT的D严格为Source正确Surface且Q90-PL新变Cloud；全池506798，非G/P内450058。T25XEG全部D206076，P内197082，两者不混用。
- outside_D_new_Surface_to_Cloud限定Source正确Surface、原Q90-PL不为Cloud、本轮变Cloud；不等于所有新增错误。outside_B_new_SourceShadow_to_Cloud限定原SourceShadow真Cloud且旧Q90非Cloud，本轮Cloud；SourceWrong列另含SourceSurface真Cloud。
- PROTECTION_COST按原影像定义P/A/其余区域及真值类逐组报告，可核查P内真Cloud/Shadow误保护；不能把P视为真Surface或参考R。
- 同512 PROBES的Q90为相同尝试步缓存。全图评价的基线Q90为固定1000步末权重。固定前三诊断产品T25XEG/T40XDR/T35RPM在全部32组表中保留；不按本轮结果重新选择。
- 曲线是单seed原始逐尝试步轨迹；45个空A步骤为缺口，SP在P为空时为0。没有平滑、跨seed置信区间或由探针选模。
''',encoding='utf-8')
    write(S/'EDITORIAL_RECEIPT.json',dict(raw_RESULT_sha256=sha(raw),published_RESULT_sha256=sha(S/'RESULT.md'),all_numeric_CSV_unchanged=True,numeric_CSV_SHA256=original_csv,raw_report_and_failure_preserved=True,visual_review=dict(layout='readable; labels/ticks uncut; legends outside data; all raw traces shown',colorblind_and_gray='solid/dashed distinction retained; separate panels; no dual axes',program_QA_no_issues=True,reviewed_files=['TRAINING_CURVES.png','TRAINING_CURVES_gray.png']),purpose='State negative aggregate tradeoff, class-union effect in largest group, nonzero SP gradients and legacy table-field meanings; no recalculation or new experimental choice'))
    p=S/'phase310.md';notes=p.read_text(encoding='utf-8')
    notes+='- 原失败：首次原304接受步AST漏绑定t/l，在正式更新前停止；仅3次门检前向、0正式更新。recovery01封存原日志/门检/初始权重并仅补原裁剪上下文，所有规则/输入/容差不变。正式训练仍只有一次955更新。\n'
    notes+='- 门检实测：零LoRA误差与初始SP均0；lambda0 loss差0、梯度最大差2.444721758365631e-9、一步更新最大差2.75605998467654e-7；旧Q90首development九窗重放误差0/pred一致。\n'
    notes+='- 资源实测：失败门检12.068秒、正式train464.114秒、evaluate66.504秒、独立CPU74.588秒；峰值CPU3462619136字节、最低宿主空闲29330292736字节；全部阶段及失败计入预算。\n'
    notes+='- 结果取舍：原D修复371012/506798，原纠正保留362132/784478；组均mIoU63.418107→57.078857%，不能只以Surface损伤修复宣称总体成功。最大组Surface IoU改善但新增Shadow导致union2→3，按原规则保留。\n'
    notes+='- 出版核对：数值CSV和原始报告/失败证据保持SHA；仅澄清总体代价、最大组缺类口径及旧字段名，曲线彩色/灰度均已人工核查。\n'
    p.write_text(notes,encoding='utf-8')
    summary=load(S/'SUMMARY_SHA256.json');summary['files']={p.name:sha(p) for p in S.iterdir() if p.is_file() and p.name!='SUMMARY_SHA256.json'};summary['code']={p.relative_to(R).as_posix():sha(p) for p in (R/'src/tools').glob('*phase310*.py')};write(S/'SUMMARY_SHA256.json',summary)
    print('Editorial review published; all raw numerical tables preserved')
if __name__=='__main__':main()
