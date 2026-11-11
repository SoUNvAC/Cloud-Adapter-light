"""Verify transferred evidence, append historical record and seal local delivery."""
from pathlib import Path
import json,hashlib,csv
ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'outputs/phase426/phase426_20261101'
def js(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 for manifest in ['TRANSFER_MANIFEST.json','REPORT_CODE_LOCK.json']:
  for n,h in js(R/manifest).items():assert sha(R/n)==h,n
 for n,h in js(R/'REPORT_COMPLETE.json')['files'].items():assert sha(R/n)==h,n
 receipt=js(R/'INDEPENDENT_ACCEPTANCE.json');complete=js(R/'COMPLETE.json')
 assert receipt['passed'] and complete['passed']
 assert receipt['calls']==complete['new_calls']==2128
 assert receipt['updates']==complete['new_updates']==2000
 end={}
 for mode in ['baseline','competitive']:
  for step in [500,1000]:
   m=js(R/mode/f'evaluations/M/step_{step}/METRICS.json')
   cm=[[sum(g['confusion'][i][j] for g in m['per_group']) for j in range(3)] for i in range(3)]
   assert len(m['per_group'])==32 and cm==m['confusion']
   ious=[100*cm[i][i]/(sum(cm[i])+sum(row[i] for row in cm)-cm[i][i]) for i in range(3)]
   assert abs(sum(ious)/3-m['mIoU'])<1e-9
   if step==1000:end[mode]=m
 delta=end['competitive']['mIoU']-end['baseline']['mIoU']
 assert abs(delta-receipt['delta_mIoU'])<1e-9
 (R/'README.md').write_bytes((R/'RESULT.md').read_bytes())
 title='## Phase426：三类语义token与竞争路由（2026-11-01）'
 entry=title+'\n\n- 完整方案直接训练，无前置特征探针；153 fit/完整32 development，seed40901，各1000步。\n'
 for mode,m in end.items():entry+=f"- {mode}：mIoU {m['mIoU']:.4f}%，三类IoU {m['IoU']}。\n"
 entry+=f'- 同轮净变化 {delta:+.4f}个百分点；新增15参数、0.05语义辅助损失及在线困难度权重，不能单独归因架构。\n- 2128次前向、2000次更新；共同初始化/逐步输入与LR/冻结/预测与AP复算通过。单seed开发集探索。\n- outputs/phase426/phase426_20261101/RESULT.md。\n\n'
 p=ROOT/'src/EXPERIMENT_LOG.md';raw=p.read_bytes();marker='## 后续维护规则'.encode();assert raw.count(marker)==1
 if title.encode() not in raw:p.write_bytes(raw.replace(marker,entry.replace('\n','\r\n').encode()+marker))
 p=ROOT/'src/research_plans/phase426.md';s=p.read_text(encoding='utf-8')
 status=f'完成：2128次前向、2000次更新，验收通过；完整改造相对同轮基线 {delta:+.4f}点。结果见 outputs/phase426/phase426_20261101/RESULT.md。'
 if status not in s:p.write_text(s+'\n'+status+'\n',encoding='utf-8')
 (R/'DELIVERY_COMPLETE.json').write_text(json.dumps({'passed':True,'calls':2128,'updates':2000,'delta_mIoU':delta,'local_CM_recomputed':True},indent=2))
 (R/'LOCAL_MANIFEST.json').write_text(json.dumps({p.relative_to(R).as_posix():sha(p) for p in R.rglob('*') if p.is_file() and p.name!='LOCAL_MANIFEST.json'},indent=2))
 print((R/'RESULT.md').read_text(encoding='utf-8'))
if __name__=='__main__':main()
