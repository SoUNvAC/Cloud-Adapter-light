"""Verify Phase427 receipts, compare fixed endpoints and seal local evidence."""
from pathlib import Path
import json,hashlib,csv
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase427/phase427_20261101'
def js(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 for manifest in ['TRANSFER_MANIFEST.json','REPORT_CODE_LOCK.json']:
  for n,h in js(R/manifest).items():assert sha(R/n)==h,n
 for n,h in js(R/'REPORT_COMPLETE.json')['files'].items():assert sha(R/n)==h,n
 receipt=js(R/'INDEPENDENT_ACCEPTANCE.json');complete=js(R/'COMPLETE.json')
 assert receipt['passed'] and complete['passed']
 assert receipt['calls']==complete['new_calls']==3192
 assert receipt['updates']==complete['new_updates']==3000
 end={};channels=[]
 for mode in ['baseline','unfrozen','integrated']:
  for step in [500,1000]:
   m=js(R/mode/f'evaluations/M/step_{step}/METRICS.json')
   cm=[[sum(g['confusion'][i][j] for g in m['per_group']) for j in range(3)] for i in range(3)]
   assert len(m['per_group'])==32 and cm==m['confusion']
   ious=[100*cm[i][i]/(sum(cm[i])+sum(row[i] for row in cm)-cm[i][i]) for i in range(3)]
   assert abs(sum(ious)/3-m['mIoU'])<1e-9
   if step==1000:
    end[mode]=m
    for g in m['per_group']:
     channels.append({'mode':mode,'product':g['product'],'surface_to_cloud':g['confusion'][0][1],'surface_to_shadow':g['confusion'][0][2],'cloud_to_surface':g['confusion'][1][0],'cloud_to_shadow':g['confusion'][1][2],'shadow_to_surface':g['confusion'][2][0],'shadow_to_cloud':g['confusion'][2][1]})
 for c in js(R/'MECHANISM_SCREEN.json')['comparisons']:
  assert abs(c['delta_mIoU']-(end[c['mode']]['mIoU']-end[c['control']]['mIoU']))<1e-9
 with (R/'SCENE_ERROR_CHANNELS.csv').open('w',newline='',encoding='utf-8') as f:
  writer=csv.DictWriter(f,fieldnames=list(channels[0]));writer.writeheader();writer.writerows(channels)
 (R/'README.md').write_bytes((R/'RESULT.md').read_bytes())
 title='## Phase427：Cloud-Adapter内融合MsRE与局部细节反馈（2026-11-01）'
 entry=title+'\n\n- 三组各1000步；153 fit/完整32 development、seed40901，原分割损失；新增较高分辨率局部读取和语义反馈，无辅助监督。\n'
 for mode,m in end.items():entry+=f"- {mode}：mIoU {m['mIoU']:.4f}%，三类IoU {m['IoU']}。\n"
 delta=end['integrated']['mIoU']-end['baseline']['mIoU'];control_delta=end['integrated']['mIoU']-end['unfrozen']['mIoU']
 entry+=f'- 融合相对基线 {delta:+.4f}点，相对仅解冻 {control_delta:+.4f}点。\n- 3192次前向、3000次更新；共同初始化、输入/LR配对、指定冻结范围、完整预测与AP复算通过。后两组允许Source交互层更新，不能称全Source冻结。单seed开发集探索。\n- outputs/phase427/phase427_20261101/RESULT.md。\n\n'
 p=ROOT/'src/EXPERIMENT_LOG.md';raw=p.read_bytes();marker='## 后续维护规则'.encode();assert raw.count(marker)==1
 if title.encode() not in raw:p.write_bytes(raw.replace(marker,entry.replace('\n','\r\n').encode()+marker))
 p=ROOT/'src/research_plans/phase427.md';s=p.read_text(encoding='utf-8')
 status=f'完成：3192次前向、3000次更新，验收通过；融合相对基线 {delta:+.4f}点，相对仅解冻 {control_delta:+.4f}点。结果见 outputs/phase427/phase427_20261101/RESULT.md。'
 if status not in s:p.write_text(s+'\n'+status+'\n',encoding='utf-8')
 (R/'DELIVERY_COMPLETE.json').write_text(json.dumps({'passed':True,'calls':3192,'updates':3000,'delta_vs_baseline':delta,'delta_vs_unfrozen':control_delta,'local_CM_recomputed':True},indent=2))
 (R/'LOCAL_MANIFEST.json').write_text(json.dumps({p.relative_to(R).as_posix():sha(p) for p in R.rglob('*') if p.is_file() and p.name!='LOCAL_MANIFEST.json'},indent=2))
 print((R/'RESULT.md').read_text(encoding='utf-8'))
if __name__=='__main__':main()
