"""Verify transferred Phase425 receipts and record scene/class comparisons."""
from pathlib import Path
import json,csv,hashlib
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase425/phase425_20261101'
def js(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 for name in ['TRANSFER_MANIFEST.json','REPORT_CODE_LOCK.json']:
  for n,h in js(R/name).items():assert sha(R/n)==h,n
 for n,h in js(R/'REPORT_COMPLETE.json')['files'].items():assert sha(R/n)==h,n
 a=js(R/'INDEPENDENT_ACCEPTANCE.json');complete=js(R/'COMPLETE.json');assert a['passed'] and complete['passed']
 assert a['calls']==complete['new_calls']==3192 and a['updates']==complete['new_updates']==3000
 rows=list(csv.DictReader((R/'TARGET_SUMMARY.csv').open()));assert len(rows)==6
 modes=['material_control','centered','centered_supported'];end={}
 for mode in modes:
  for step in [500,1000]:
   m=js(R/mode/f'evaluations/M/step_{step}/METRICS.json');cm=[[sum(g['confusion'][i][j] for g in m['per_group']) for j in range(3)] for i in range(3)];assert len(m['per_group'])==32 and cm==m['confusion']
   value=sum(100*cm[i][i]/(sum(cm[i])+sum(r[i] for r in cm)-cm[i][i]) for i in range(3))/3
   rr=next(r for r in rows if r['mode']==mode and r['step']==str(step));assert abs(value-float(rr['mIoU']))<1e-9
   if step==1000:end[mode]=m
 baseline=js(ROOT/'outputs/phase421/phase421_20261101/baseline/evaluations/M/step_1000/METRICS.json')
 screen=js(R/'MECHANISM_SCREEN.json')
 reference411=js(R/'PROMOTION.json')['baseline_1000']
 for mode,paired in [('centered',['material_control']),('centered_supported',['material_control','centered'])]:
  margin=end[mode]['mIoU']-max([baseline['mIoU'],reference411]+[end[c]['mIoU'] for c in paired]);assert abs(margin-screen['mechanisms'][mode]['margin_pp'])<1e-9
 scene=[]
 for mode,m in end.items():
  for g in m['per_group']:
   scene.append({'mode':mode,'product':g['product'],'surface_to_cloud':g['confusion'][0][1],'surface_to_shadow':g['confusion'][0][2],'cloud_to_surface':g['confusion'][1][0],'cloud_to_shadow':g['confusion'][1][2],'shadow_to_surface':g['confusion'][2][0],'shadow_to_cloud':g['confusion'][2][1]})
 with (R/'SCENE_ERROR_CHANNELS.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(scene[0]));w.writeheader();w.writerows(scene)
 lines=['# Phase425：约束是否减少损伤','','原标签与全部32组均保留。以下同一高亮场景只作诊断，主指标仍为完整目标集。','', '|方案|T25XEG地表→云|','|---|---:|']
 for row in scene:
  if 'T25XEG' in row['product']:lines.append(f"|{row['mode']}|{row['surface_to_cloud']}|")
 lines+=['',f"独立MsRE基线mIoU {baseline['mIoU']:.4f}%；Phase411参考 {reference411:.4f}%。",'', '|方案|mIoU相对Phase421基线|地表IoU变化|云IoU变化|阴影IoU变化|','|---|---:|---:|---:|---:|']
 for mode,m in end.items():lines.append(f"|{mode}|{m['mIoU']-baseline['mIoU']:+.4f}|"+'|'.join(f"{m['IoU'][i]-baseline['IoU'][i]:+.4f}" for i in range(3))+'|')
 lines+=['','归零约束取消输出层偏置的功能，三组虽具有同样数量的参数张量，实际有效自由度并非完全相同。亮度支持只是预先固定的代理条件。单seed只能评价本次实现，不能据此宣布泛化或物理机制成立。']
 (R/'INTERPRETATION.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 (R/'README.md').write_bytes((R/'RESULT.md').read_bytes())
 title='## Phase425：材质残差归零与亮度支持约束（2026-11-01）'
 entry=title+'\n\n- 三组各1000步、seed40901、原损失与完整32组；不加末端修正，不改标签。\n'
 for mode,m in end.items():entry+=f"- {mode}：mIoU {m['mIoU']:.4f}%，相对Phase421独立MsRE基线 {m['mIoU']-baseline['mIoU']:+.4f}点。\n"
 entry+='- 3192次前向、3000次更新；归零/门控/初始化测试、逐步输入与LR配对、冻结及本地指标复核通过。归零组384个输出bias代数抵消。单seed筛选，不自动延长。\n- outputs/phase425/phase425_20261101/RESULT.md。\n\n'
 log=ROOT/'src/EXPERIMENT_LOG.md';raw=log.read_bytes();marker='## 后续维护规则'.encode();assert raw.count(marker)==1
 if title.encode() not in raw:log.write_bytes(raw.replace(marker,entry.replace('\n','\r\n').encode()+marker))
 plan=ROOT/'src/research_plans/phase425.md';s=plan.read_text(encoding='utf-8');status='完成：三组3192次前向、3000次更新，验收通过。结果见 outputs/phase425/phase425_20261101/RESULT.md。'
 if status not in s:plan.write_text(s+'\n'+status+'\n',encoding='utf-8')
 (R/'DELIVERY_COMPLETE.json').write_text(json.dumps({'passed':True,'calls':3192,'updates':3000,'local_CM_recomputed':True},indent=2))
 manifest={p.relative_to(R).as_posix():sha(p) for p in R.rglob('*') if p.is_file() and p.name!='LOCAL_MANIFEST.json'}
 (R/'LOCAL_MANIFEST.json').write_text(json.dumps(manifest,indent=2));print('sealed',len(manifest),'files')
if __name__=='__main__':main()
