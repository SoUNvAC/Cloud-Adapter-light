"""Seal the protocol audit appendix without rewriting historical phase deliveries."""
from pathlib import Path
import json,hashlib,re
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/audit_phase400_427/20261101'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 checks=json.loads((R/'AUDIT_CHECKS.json').read_text(encoding='utf-8'))
 assert checks['phases']==28 and checks['formal_trace_rows']==82000
 assert not any(checks[k] for k in ['noncontinuous','nonfinite','paired_failures','manifest_mismatches','manifest_missing'])
 assert all(x['paired_to405'] for x in checks['H_pairs_vs405'])
 p=R/'RESULT.md';s=p.read_text(encoding='utf-8')
 assert set(map(int,re.findall(r'^\|(4\d\d)\|',s,re.M)))==set(range(400,428))
 for name in ['INVENTORY.json','TRAINING_RUNS.csv','AUDIT_CHECKS.json','RECEIPTS.json','EVIDENCE_SHA256.json']:
  s=s.replace(f']({name})',f']({(R/name).as_posix()})')
 p.write_text(s,encoding='utf-8')
 log=ROOT/'src/EXPERIMENT_LOG.md';raw=log.read_bytes();marker='## 后续维护规则'.encode();assert raw.count(marker)==1
 title='## Phase400—427训练协议审计附录（2026-11-01）'
 entry=title+'\n\n- 仅审计，零训练/前向；不改旧Phase封存结果及标签。覆盖28轮，本地49份正式轨迹82000行、3877项清单哈希复核。\n- 413—415辅助优化混杂已有416—418纠正；412起大多1000步筛选，不能代表充分收敛上限；426结构与辅助监督未拆分；427只验证局部适配，不能否定完整融合网络重训。\n- 早期源域保持判据与当前目标优先不同，406全强度H目标收益作为候选保留。完整逐轮保留/限定/复验建议见 outputs/audit_phase400_427/20261101/RESULT.md。\n- 本次未重载远端权重/像元数组，历史数值验收与本次协议审计范围分开说明。\n\n'
 if title.encode() not in raw:log.write_bytes(raw.replace(marker,entry.replace('\n','\r\n').encode()+marker))
 evidence=json.loads((R/'EVIDENCE_SHA256.json').read_text(encoding='utf-8'))
 evidence['src/tools/finalize_audit_phase400_427.py']=sha(Path(__file__))
 (R/'EVIDENCE_SHA256.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
 (R/'DELIVERY_COMPLETE.json').write_text(json.dumps({'passed':True,'scope':'protocol audit, not new full numerical validation','phases':28,'new_updates':0,'new_forwards':0},indent=2),encoding='utf-8')
 manifest={p.name:sha(p) for p in R.iterdir() if p.is_file() and p.name!='LOCAL_MANIFEST.json'}
 (R/'LOCAL_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
 assert all(sha(R/n)==h for n,h in manifest.items())
 print('Audit sealed:',len(manifest),'files; all 28 phase rows present; no training launched.')
if __name__=='__main__':main()
