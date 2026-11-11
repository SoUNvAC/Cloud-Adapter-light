"""Local independent recomputation and evidence sealing for Phase421."""
from pathlib import Path
import json,csv,hashlib
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase421/phase421_20261101'
def js(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 for name in ['TRANSFER_MANIFEST.json','REPORT_CODE_LOCK.json']:
  for n,h in js(R/name).items():assert sha(R/n)==h,n
 for n,h in js(R/'REPORT_COMPLETE.json')['files'].items():assert sha(R/n)==h,n
 a=js(R/'INDEPENDENT_ACCEPTANCE.json');done=js(R/'COMPLETE.json')
 assert a['passed'] and done['passed']
 assert a['calls']==done['new_calls']==3192 and a['updates']==done['new_updates']==3000
 assert a['label_arrays_exact']==192
 rows=list(csv.DictReader((R/'TARGET_SUMMARY.csv').open()));assert len(rows)==6
 for mode in ['baseline','neighborhood','material']:
  for step in [500,1000]:
   m=js(R/mode/f'evaluations/M/step_{step}/METRICS.json');cm=[[sum(g['confusion'][i][j] for g in m['per_group']) for j in range(3)] for i in range(3)]
   assert len(m['per_group'])==32 and cm==m['confusion']
   ii=[100*cm[i][i]/(sum(cm[i])+sum(r[i] for r in cm)-cm[i][i]) for i in range(3)]
   row=next(r for r in rows if r['mode']==mode and r['step']==str(step))
   assert abs(sum(ii)/3-float(row['mIoU']))<1e-9
 end={r['mode']:r for r in rows if r['step']=='1000'};screen=js(R/'MECHANISM_SCREEN.json')
 margin=float(end['material']['mIoU'])-max(float(end[c]['mIoU']) for c in ['baseline','neighborhood'])
 assert abs(margin-screen['margin_pp'])<1e-9
 title='## Phase421：MsRE局部材质参考三组对照（2026-11-01）'
 entry=title+'\n\n- 三组各1000步、seed40901、153 fit/32 development；原始Source起步，不用Phase420修正，不加辅助损失。\n'
 for mode,v in end.items():entry+=f"- {mode}：mIoU {float(v['mIoU']):.4f}、影IoU {float(v['Shadow_IoU']):.4f}、影宏AP {float(v['macro_AP']):.4f}。\n"
 entry+=f"- 材质版相对最强对照 {margin:+.4f}个百分点；超过两对照：{screen['beats_both']}。单seed筛选，不自动追加训练。\n- 3192次前向、3000次更新，配对/冻结/完整预测重算/原标签一致验收通过。\n- outputs/phase421/phase421_20261101/RESULT.md。\n\n"
 log=ROOT/'src/EXPERIMENT_LOG.md';raw=log.read_bytes();marker='## 后续维护规则'.encode();assert raw.count(marker)==1
 if title.encode() not in raw:log.write_bytes(raw.replace(marker,entry.replace('\n','\r\n').encode()+marker))
 p=ROOT/'src/research_plans/phase421.md';s=p.read_text(encoding='utf-8');status='完成：三组3192次前向、3000次更新，验收通过。结果见 outputs/phase421/phase421_20261101/RESULT.md。'
 if status not in s:p.write_text(s+'\n'+status+'\n',encoding='utf-8')
 (R/'README.md').write_bytes((R/'RESULT.md').read_bytes())
 (R/'DELIVERY_COMPLETE.json').write_text(json.dumps({'passed':True,'calls':3192,'updates':3000,'local_metrics_recomputed':True,'margin_pp':margin},indent=2))
 manifest={p.relative_to(R).as_posix():sha(p) for p in R.rglob('*') if p.is_file() and p.name!='LOCAL_MANIFEST.json'}
 (R/'LOCAL_MANIFEST.json').write_text(json.dumps(manifest,indent=2));print('sealed',len(manifest),'files')
if __name__=='__main__':main()
