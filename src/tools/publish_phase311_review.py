"""Local editorial publication and file-SHA only; no scientific recomputation."""
from phase311_common import *

def main():
    root=LOCAL/'remote_backup';b=load(LOCAL/'BACKUP_RECEIPT.json')
    assert b['all_archive_members_verified'] and b['archive_SHA_verified_before_extraction']
    original=load(root/'REVIEW_ACCEPTANCE.json');assert original['status']=='passed'
    assert sha(root/'RESULT.md')==original['reviewed_report_sha256']
    csv_sha={p.name:sha(p) for p in S.glob('*.csv')}
    assert all(sha(root/n)==h for n,h in csv_sha.items())
    report=(root/'RESULT.md').read_text(encoding='utf-8')
    report=report.replace('原45空A步骤不前向；初始CM≤1e−10、原304 AST lambda0 loss/梯度/更新≤1e−6通过。\n|策略|', '原45空A步骤不前向；初始CM≤1e−10、原304 AST lambda0 loss/梯度/更新≤1e−6通过。\n\n|策略|')
    report=report.replace('CM固定末步32×9窗。\n|策略|', 'CM固定末步32×9窗。\n\n|策略|')
    report+='\n损伤转移与误保护：相对PL，Surface→Cloud净减少475251，但Cloud→Surface增加166987、Cloud→Shadow增加625324、Surface→Shadow增加48069。P内真Cloud208670中186722判为Surface，PL原正确被CM改错70168；P内真Shadow94432中83449判为Surface，PL原正确被CM改错2668。这些“改错”数包括全部新错误类别，不全等于转Surface；P内Shadow同时有7782从PL错改对，不应概括为各类均恶化。\n'
    report+='\n本轮未取得优于PL的总体取舍，也未实现比SP保留更多原Cloud纠正；最大组虽已完全修复，不能代替全32组结论。本保护区连续换loss分枝在此记录收束，无新证据不再追加loss或lambda扫描；下一步须另提假设并获得新的执行指令。\n'
    report+='\n曲线视觉复核：少数步骤的CM与梯度尖峰原样保留，不平滑、不截断，也未事后增加梯度裁剪；均值不能代替逐步活动表，不将本次曲线概括为稳定收敛。\n'
    (S/'RESULT.md').write_text(report,encoding='utf-8')
    notes=(S/'phase311.md').read_text(encoding='utf-8').split('- 交付复核：',1)[0]
    notes+='- 交付复核：组均mIoU CM61.5651%、PL63.4181%、SP57.0789%；CM池化72.5145%。最大T25XEG三类误报清零并恢复100%，但原Cloud纠正仅保留299949/784478，比SP少62183；总体取舍未改善。\n'
    notes+='- 实际资源：GPU训练/评价共542.2573秒；独立CPU峰3664150528字节，宿主最低可用28944883712字节，均在止损线内。共2238次前向，无正式实验失败或恢复；CuBLAS历史非确定性警告及可选模块导入提示原样保留，未修改环境。\n'
    notes+='- 口径：四策略32组、完整27格及独立CSV边际复算、固定D/B、原最大组206076与P子集197082、192探针行、390活动步和四梯度审计均通过。报告仅澄清实际代价，数值CSV与全量备份原件不改。\n'
    (S/'phase311.md').write_text(notes,encoding='utf-8')
    assert all(sha(S/n)==h for n,h in csv_sha.items())
    write(S/'EDITORIAL_RECEIPT.json',dict(raw_reviewed_report_sha256=sha(root/'RESULT.md'),published_report_sha256=sha(S/'RESULT.md'),all_numeric_CSV_unchanged=csv_sha,clarifications='Actual costs, branch conclusion, blank lines for tables; no metric or model changes',visual_review_PNG_and_grayscale='Passed: three ordered unsmoothed panels, legends and labels clear, 45 skipped-step gaps, no clipping or overlap; single-run diagnostic, no uncertainty claim.'))
    write(S/'SUMMARY_SHA256.json',dict(status='completed',files={p.name:sha(p) for p in S.iterdir() if p.is_file() and p.name!='SUMMARY_SHA256.json'},code={p.relative_to(R).as_posix():sha(p) for p in (R/'src/tools').glob('*phase311*.py')}))
    print('Editorial publication complete; all numeric CSV bytes preserved')

if __name__=='__main__':main()
