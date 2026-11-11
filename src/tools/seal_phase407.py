"""Seal local delivery only after successful independent acceptance."""
from pathlib import Path
import json, csv, hashlib
ROOT = Path(__file__).resolve().parents[2]
R = ROOT/'outputs/phase407/phase407_20261101'

def main():
    acceptance = json.loads((R/'INDEPENDENT_ACCEPTANCE.json').read_text())
    complete = json.loads((R/'COMPLETE.json').read_text())
    assert acceptance['passed'] and complete['passed']
    with (R/'paired_deltas.csv').open() as f:
        paired=list(csv.DictReader(f))
    entries=[f"seed{v['seed']}：406−405匹配统一缩放目标差{float(v['target_delta_406_minus_405']):+.4f}、源差{float(v['source_delta_406_minus_405']):+.4f}；406−原weak_5源差{float(v['source_delta_406_minus_weak5']):+.4f}个百分点。" for v in paired]
    (R/'README.md').write_bytes((R/'RESULT.md').read_bytes())
    status = f"完成：2026-11-01，3420次前向、零训练，{complete['seconds']/60:.2f}分钟，独立验收通过。结果见 outputs/phase407/phase407_20261101/RESULT.md。未启动408。"
    plan = ROOT/'src/research_plans/phase407.md'
    text = plan.read_text(encoding='utf-8')
    if status not in text:
        plan.write_text(text+'\n'+status+'\n',encoding='utf-8')
    (R/'HANDOFF.md').write_text('# Phase407 已完成\n\n'+status+'\n权重仍在原 /home/scv/shared/phase405_20261101 与 phase406_20261101；本轮未产生新权重，无需重启。\n',encoding='utf-8')
    log=ROOT/'src/EXPERIMENT_LOG.md';raw=log.read_bytes();marker='## 后续维护规则'.encode()
    title='## Phase407：目标精度对齐后的源域比较（2026-11-01）'
    assert raw.count(marker)==1
    if title.encode() not in raw:
        entry=title+'\n\n- 固定405/406两个seed最终4000步权重，3420前向、零更新；32 development选统一alpha，选定后535源val评价。每模型8次中点探测；目标参照为原405 weak_5，容差0.05点；不追加搜索。\n'
        entry+=''.join('- '+v+'\n' for v in entries)
        entry+='- 主要判据'+('通过' if acceptance['scientific_criterion_passed'] else '未通过')+'；完整解释见RESULT。目标development参与匹配、源val前轮已暴露，不作独立泛化或显著性证明。\n'
        entry+=f"- 控制器{complete['seconds']:.1f}秒。输入/权重/代码/传输身份、8步目标选点重放、参考与选点预测复现、调用账本及逐图汇总指标验收通过；新点池化AP未独立像元复算。\n"
        entry+='- 交付outputs/phase407/phase407_20261101/RESULT.md、matched_summary.csv、paired_deltas.csv及完整小型证据。未启动408。\n\n'
        if not acceptance['scientific_criterion_passed'] and all(float(v['source_delta_406_minus_weak5'])<0 for v in paired):
            entry+='决策：暂停当前全层能量惩罚及细搜，保留原weak_5固定基线，后续优先考虑分层训练设计；不推广为所有正则无效。\n\n'
        log.write_bytes(raw.replace(marker,entry.replace('\n','\r\n').encode()+marker))
    (R/'DELIVERY_COMPLETE.json').write_text(json.dumps({'passed':True,'phase':407,'calls':3420,'updates':0,'independent_acceptance':True,'scientific_criterion_passed':acceptance['scientific_criterion_passed'],'next_phase_started':False},indent=2))
    code = {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'src/tools').glob('*phase407*.py'))}
    (R/'LOCAL_CODE_LOCK.json').write_text(json.dumps(code,indent=2))
    manifest={p.relative_to(R).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(R.rglob('*')) if p.is_file() and p.name!='LOCAL_MANIFEST.json'}
    (R/'LOCAL_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    print('Phase407 delivery sealed:',len(manifest),'files')

if __name__=='__main__':main()
