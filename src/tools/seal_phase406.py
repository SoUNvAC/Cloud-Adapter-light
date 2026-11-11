"""Seal local delivery only after successful independent acceptance."""
from pathlib import Path
import json, csv, hashlib
ROOT = Path(__file__).resolve().parents[2]
R = ROOT/'outputs/phase406/phase406_20261101'

def main():
    acceptance = json.loads((R/'INDEPENDENT_ACCEPTANCE.json').read_text())
    complete = json.loads((R/'COMPLETE.json').read_text())
    assert acceptance['passed'] and complete['passed']
    with (R/'versus_phase405.csv').open() as f:
        paired = list(csv.DictReader(f))
    entries = []
    for case in ['weak_5','1.0']:
        values = [v for v in paired if v['case']==case]
        entries.append(case+'目标/源差：'+'；'.join(f"seed{v['seed']} {float(v['target_mIoU']):+.4f}/{float(v['source_mIoU']):+.4f}" for v in values)+'个百分点（同seed同缩放对405）。')
    (R/'README.md').write_bytes((R/'RESULT.md').read_bytes())
    status = f"完成：2026-11-01，15620次前向、8240次更新，{complete['seconds']/60:.2f}分钟，独立验收通过。结果见 outputs/phase406/phase406_20261101/RESULT.md。未启动407。"
    plan = ROOT/'src/research_plans/phase406.md'
    text = plan.read_text(encoding='utf-8')
    if status not in text:
        plan.write_text(text+'\n'+status+'\n',encoding='utf-8')
    (R/'HANDOFF.md').write_text('# Phase406 已完成\n\n'+status+'\n远端权重保留于 /home/scv/shared/phase406_20261101，无需重启。\n',encoding='utf-8')
    log=ROOT/'src/EXPERIMENT_LOG.md';raw=log.read_bytes();marker='## 后续维护规则'.encode()
    title='## Phase406：训练期残差能量约束（2026-11-01）'
    assert raw.count(marker)==1
    if title.encode() not in raw:
        entry=title+'\n\n- 固定系数1的四层归一化残差能量惩罚；H26144参数，Source冻结。复用405两个seed、初始化、逐步输入/增强/学习率，固定最终4000步。153 fit训练，32 development与535源val仅评价。\n'
        entry+=''.join('- '+v+'\n' for v in entries)
        entry+='- 主要weak_5双域改善判据'+('通过' if acceptance['scientific_dual_domain_criterion_passed'] else '未通过')+'。完整解释见RESULT；两seed及已暴露数据仅作探索。\n'
        entry+=f"- 15620次完整前向、8240次更新（8000正式+240短跑），{complete['seconds']:.1f}秒控制器墙钟。参数/冻结/调用/输入身份、逐图汇总指标和端点验收通过；正则梯度自测与实际损失通过；新点池化AP未独立像元复算。\n"
        entry+='- 交付outputs/phase406/phase406_20261101/RESULT.md、versus_phase405.csv、penalty_summary.csv及原始证据。未启动407。\n\n'
        log.write_bytes(raw.replace(marker,entry.replace('\n','\r\n').encode()+marker))
    (R/'DELIVERY_COMPLETE.json').write_text(json.dumps({'passed':True,'phase':406,'calls':15620,'updates':8240,'independent_acceptance':True,'scientific_dual_domain_criterion_passed':acceptance['scientific_dual_domain_criterion_passed'],'next_phase_started':False},indent=2))
    code = {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'src/tools').glob('*phase406*.py'))}
    (R/'LOCAL_CODE_LOCK.json').write_text(json.dumps(code,indent=2))
    manifest={p.relative_to(R).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(R.rglob('*')) if p.is_file() and p.name!='LOCAL_MANIFEST.json'}
    (R/'LOCAL_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    print('Phase406 delivery sealed:',len(manifest),'files')

if __name__=='__main__':main()
