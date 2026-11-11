"""Local metric recomputation and strength-scan delivery."""
from pathlib import Path
import json,hashlib,csv
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase423/phase423_20261101'
def js(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 for n,h in js(R/'TRANSFER_MANIFEST.json').items():assert sha(R/n)==h
 for n,h in js(R/'COMPLETE.json')['files'].items():assert sha(R/n)==h
 a=js(R/'ACCEPTANCE.json');assert a['passed'] and a['endpoint_exact_scenes']==64 and a['label_arrays_exact']==160
 assert len((R/'CALL_LEDGER.jsonl').read_text().splitlines())==160 and a['updates']==0
 rows=[];scenes=[]
 for alpha in [0.,.25,.5,.75,1.]:
  m=js(R/f'alpha_{alpha:g}'/'METRICS.json');assert len(m['per_group'])==32
  cm=[[sum(g['confusion'][i][j] for g in m['per_group']) for j in range(3)] for i in range(3)];assert cm==m['confusion']
  v=sum(100*cm[i][i]/(sum(cm[i])+sum(r[i] for r in cm)-cm[i][i]) for i in range(3))/3;assert abs(v-m['mIoU'])<1e-9
  rows.append(m)
  for g in m['per_group']:
   scenes.append({'alpha':alpha,'product':g['product'],'surface_to_cloud':g['confusion'][0][1],'surface_to_shadow':g['confusion'][0][2],'cloud_to_shadow':g['confusion'][1][2],'shadow_to_surface':g['confusion'][2][0],'shadow_to_cloud':g['confusion'][2][1],'fixes_vs_on':g['fixes_vs_on'],'regressions_vs_on':g['regressions_vs_on']})
 with (R/'SCENE_ERROR_CHANNELS.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(scenes[0]));w.writeheader();w.writerows(scenes)
 screen=js(R/'SCREEN_RESULT.json');best=max(rows,key=lambda r:(r['mIoU'],-r['alpha']));assert best['alpha']==screen['best_alpha']
 critical=[s for s in scenes if 'T25XEG' in s['product']]
 lines=['# Phase423 诊断解释','','高亮低纹理场景T25XEG的地表→云误报（全部场景仍计入主指标）：','', '|alpha|地表→云像素|','|---|---:|']
 for s in critical:lines.append(f"|{s['alpha']:g}|{s['surface_to_cloud']}|")
 lines+=['',f"预设网格最佳alpha={best['alpha']:g}，mIoU={best['mIoU']:.4f}%，相对独立基线{screen['gain_vs_baseline_pp']:+.4f}点，相对关闭{screen['gain_vs_off_pp']:+.4f}点。",'', '扫描只能检验一个全局残差系数是否足够，不能否定空间选择性或其他材质关系结构，也不能证明参考为何出错。所有选择在已暴露development上进行，不是独立确认。']
 (R/'INTERPRETATION.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 (R/'README.md').write_bytes((R/'RESULT.md').read_bytes())
 title='## Phase423：材质残差零训练强度扫描（2026-11-01）'
 entry=title+'\n\n- 同一Phase421材质检查点，新增残差alpha=0/0.25/0.5/0.75/1；原MsRE与标签不变。\n'
 for r in rows:entry+=f"- alpha={r['alpha']:g}：mIoU {r['mIoU']:.4f}%，影IoU {r['Shadow_IoU']:.4f}%，影宏AP {r['macro_AP']:.4f}%。\n"
 entry+=f"- 160次前向、零训练，两端点预测和分数逐位复现；开发集网格最佳相对独立基线 {screen['gain_vs_baseline_pp']:+.4f}点，不能当作独立确认。\n- outputs/phase423/phase423_20261101/RESULT.md。\n\n"
 log=ROOT/'src/EXPERIMENT_LOG.md';raw=log.read_bytes();marker='## 后续维护规则'.encode();assert raw.count(marker)==1
 if title.encode() not in raw:log.write_bytes(raw.replace(marker,entry.replace('\n','\r\n').encode()+marker))
 plan=ROOT/'src/research_plans/phase423.md';s=plan.read_text(encoding='utf-8');status='完成：160次前向、零训练，两端点逐位复现和本地指标复算通过。结果见 outputs/phase423/phase423_20261101/RESULT.md。'
 if status not in s:plan.write_text(s+'\n'+status+'\n',encoding='utf-8')
 (R/'DELIVERY_COMPLETE.json').write_text(json.dumps({'passed':True,'calls':160,'updates':0,'local_CM_recomputed':True},indent=2))
 manifest={p.relative_to(R).as_posix():sha(p) for p in R.rglob('*') if p.is_file() and p.name!='LOCAL_MANIFEST.json'}
 (R/'LOCAL_MANIFEST.json').write_text(json.dumps(manifest,indent=2));print('sealed',len(manifest),'files')
if __name__=='__main__':main()
