"""Editorial file publication only; scientific CSV, parameters and raw backup stay unchanged."""
from phase312_common import *

def main():
    root=LOCAL/'remote_backup';b=load(LOCAL/'BACKUP_RECEIPT.json');assert b['all_archive_members_verified']
    audit=load(root/'AUDIT_ACCEPTANCE.json');assert audit['status']=='passed' and audit['fit_calls_added']==audit['neural_calls']==0
    csv_sha={p.relative_to(S).as_posix():sha(p) for p in S.rglob('*.csv')}
    assert all(sha(root/n)==h for n,h in csv_sha.items())
    report=(root/'RESULT.md').read_text(encoding='utf-8')
    report+='\n本次结论是有限的固定操作点选择性增量：MS6相对同参数RGB6Q自身0.90的组均mIoU仅高0.014029pp，相对PL仅高0.006568pp。在7089个影像选择的等数量条件下，MS6比RGB6Q多修复972个D、少撤销975个K，但比Source q规则多173−11=162个Shadow错误转移。对RGB3等数量规则，优势更小；组内平均AUC也未超过RGB3（0.711220对0.711667）。不能概括为光谱排序全面优胜。\n'
    report+='\n最大T25XEG修复0，原206076损伤全部余留；约99.8%的MS6修复集中在前三组，未解决最大亮地表损伤。该固定伪标签方案的总体收益很小，不支持宣称云雪可分、跨场景泛化或可部署收益；不追加拟合或阈值搜索。\n'
    (S/'RESULT.md').write_text(report,encoding='utf-8')
    state=load(root/'STATUS.json')
    notes=(S/'phase312.md').read_text(encoding='utf-8').split('- 最终复核：',1)[0]
    notes+='- 最终复核：97个非空fit产品、794624样本、3模型均成功；97组等权，3空产品保留不补齐；32组预测/读出封存后才读本轮真值，100产品采样内标签审计通过。\n'
    notes+='- 实际取舍：MS6组均mIoU63.4247%，PL63.4181%，仅+0.006568pp；MS6修复D5863、撤销正确Cloud1053、Shadow错误转移173；最大T25XEG修复0。不以原B784478天然全保留或Shadow不变作为新增收益。\n'
    notes+=f"- 资源：原始CPU管线{state['total_seconds']:.4f}秒，独立支持/账本复核累计{audit['total_science_plus_audit_seconds']:.4f}秒；原管线峰值3280707584字节、最低宿主可用27991314432字节；预算内，零神经调用。\n"
    notes+='- 网络与失败：首次SSH拉取握手被远端关闭；确认本轮未创建后补拉取成功。首次SCP在79151104字节停滞超过6分钟；磁盘充足、新SSH可用，底层原因未确定。保留原partial与失败记录，先核前缀SHA，再从同一归档仅补尾部，整包/全成员验收通过。无科学实验失败、重拟合或重新归档。\n'
    notes+='- 求解器口径：三模型均以固定ftol判据成功停止，最终梯度无穷范数略高于gtol=1e-7；原始message/梯度/迭代保留，不宣称全部达到gtol，不追加迭代。\n'
    notes+='- 审计口径：AUDIT_ACCEPTANCE的reference_or_student_read=False指未读取新的原始真值或学生缓存；独立账本核对只读取封存后的结果表。此审计未重新拟合、未改采样/参数/预测/阈值。\n'
    (S/'phase312.md').write_text(notes,encoding='utf-8')
    import shutil
    for name in ('BACKUP_FAILURE.json','STALL_DIAGNOSIS.json','BACKUP_RECOVERY01.json'):
        shutil.copyfile(LOCAL/name,S/name)
    assert all(sha(S/n)==h for n,h in csv_sha.items())
    write(S/'EDITORIAL_RECEIPT.json',dict(raw_report_SHA=sha(root/'RESULT.md'),published_report_SHA=sha(S/'RESULT.md'),all_numeric_CSV_unchanged=csv_sha,models_unchanged={n:sha(S/'models'/(n+'.json')) for n in MODELS},changes='Interpretation, network event and stopping-criterion clarification; no metric, fit or prediction changes'))
    write(S/'SUMMARY_SHA256.json',dict(status=state['status'],scientific_status=state['scientific_status'],files={p.relative_to(S).as_posix():sha(p) for p in S.rglob('*') if p.is_file() and p.name!='SUMMARY_SHA256.json'},code={p.relative_to(R).as_posix():sha(p) for p in (R/'src/tools').glob('*phase312*.py')}))
    print('Phase312 editorial publication complete; all scientific artifacts retained')

if __name__=='__main__':main()
