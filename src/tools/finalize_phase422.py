"""Independent local CM audit and concise damage interpretation."""
from pathlib import Path
import json,hashlib,csv
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase422/phase422_20261101'
def js(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 for n,h in js(R/'TRANSFER_MANIFEST.json').items():assert sha(R/n)==h,n
 for n,h in js(R/'COMPLETE.json')['files'].items():assert sha(R/n)==h,n
 a=js(R/'ACCEPTANCE.json');assert a['passed'] and a['max_shadow_score_error']==0 and a['on_prediction_replay_exact_scenes']==32
 assert len((R/'CALL_LEDGER.jsonl').read_text().splitlines())==64 and a['updates']==0
 data={k:js(R/k/'METRICS.json') for k in ['on','off']}
 for m in data.values():
  assert len(m['per_group'])==32
  cm=[[sum(g['confusion'][i][j] for g in m['per_group']) for j in range(3)] for i in range(3)]
  assert cm==m['confusion']
  value=sum(100*cm[i][i]/(sum(cm[i])+sum(r[i] for r in cm)-cm[i][i]) for i in range(3))/3
  assert abs(value-m['mIoU'])<1e-9
 base=js(ROOT/'outputs/phase421/phase421_20261101/baseline/evaluations/M/step_1000/METRICS.json')
 on,off=data['on'],data['off'];d=js(R/'DECOMPOSITION.json')
 assert abs(d['total_on_minus_baseline_pp']-d['direct_branch_on_minus_off_pp']-d['jointly_trained_MsRE_off_minus_baseline_pp'])<1e-9
 rows=[];onmap={g['product']:g for g in on['per_group']};basemap={g['product']:g for g in base['per_group']}
 for g in off['per_group']:
  b=basemap[g['product']];o=onmap[g['product']]
  row={'product':g['product'],'fixes_after_off':g['fixes_vs_on'],'regressions_after_off':g['regressions_vs_on']}
  for i,gt in enumerate(['surface','cloud','shadow']):
   for j,pred in enumerate(['surface','cloud','shadow']):
    if i!=j:
     for state,cm in [('baseline',b['confusion']),('on',o['confusion']),('off',g['confusion'])]:row[f'{state}_{gt}_to_{pred}']=cm[i][j]
  rows.append(row)
 with (R/'SCENE_ERROR_CHANNELS.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 critical=next(x for x in rows if 'T25XEG' in x['product']);retired=js(R/'RETIRED_ATTEMPT.json')
 lines=['# Phase422 损伤解释','','新增分支的当前直接作用是本轮整体mIoU下降的主要来源；共同训练的MsRE仍有局部损伤，不能说它完全没有被影响。','',f"开启 {on['mIoU']:.4f}%，关闭 {off['mIoU']:.4f}%，独立基线 {base['mIoU']:.4f}%。关闭恢复 {off['mIoU']-on['mIoU']:.4f}个百分点。",'', '|状态|地表IoU|云IoU|阴影IoU|','|---|---:|---:|---:|']
 for name,m in [('baseline',base),('on',on),('off',off)]:lines.append('|'+name+'|'+'|'.join(f'{v:.4f}' for v in m['IoU'])+'|')
 lines+=['',f"高亮低纹理场景T25XEG，地表误判为云：基线 {critical['baseline_surface_to_cloud']}；开启 {critical['on_surface_to_cloud']}；关闭 {critical['off_surface_to_cloud']}。",'',f"关闭分支相对开启修复 {sum(x['fixes_after_off'] for x in rows)} 像素，新增错误 {sum(x['regressions_after_off'] for x in rows)} 像素。",'',f"阴影宏AP：开启 {on['macro_AP']:.4f}%，关闭 {off['macro_AP']:.4f}%。分支直接作用没有提高宏AP；其阴影IoU提升也不能等价为排序能力提升。",'', '后续应优先检验新增内部残差的触发范围、参考可靠性与幅度，而不是继续扩大这个未经限制的分支。单次开关不能证明色度/纹理先验为何失效，也不能据此认定所有损伤都来自同一种机制。','',f"执行记录：正式64次完整输入前向、零更新、2次小张量MsRE测试；首次失败尝试另有{retired['calls']}次完整输入前向和2次小张量测试。首次inference_mode/全部关闭梯度标志的执行组合未通过分数复现，保留归档；恢复原验证no_grad及原标志后32组预测与分数逐位一致。未单独分离两项执行设置的影响。"]
 (R/'INTERPRETATION.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 (R/'README.md').write_bytes((R/'RESULT.md').read_bytes())
 title='## Phase422：材质参考分支开关损伤诊断（2026-11-01）'
 entry=title+f"\n\n- 同一Phase421材质检查点，开启{on['mIoU']:.4f}%、关闭{off['mIoU']:.4f}%，独立基线{base['mIoU']:.4f}%。关闭仅绕过新增路径，保留已共同训练的MsRE。\n- 主要整体损失来自当前分支直接作用；共同训练仍留有高亮场景局部损伤。不能把端点差值当成完整训练因果分解。\n- 正式64次前向零训练，原预测/分数精确复现、权重与标签不变、独立混淆矩阵复核通过；失败尝试另计2次前向。\n- outputs/phase422/phase422_20261101/RESULT.md 与 INTERPRETATION.md。\n\n"
 log=ROOT/'src/EXPERIMENT_LOG.md';raw=log.read_bytes();marker='## 后续维护规则'.encode();assert raw.count(marker)==1
 if title.encode() not in raw:log.write_bytes(raw.replace(marker,entry.replace('\n','\r\n').encode()+marker))
 plan=ROOT/'src/research_plans/phase422.md';s=plan.read_text(encoding='utf-8');status='完成：正式64次前向零训练，开启端点逐位复现，关闭结果77.2775%；失败尝试2次前向另计，详见结果目录。'
 if status not in s:plan.write_text(s+'\n'+status+'\n',encoding='utf-8')
 (R/'DELIVERY_COMPLETE.json').write_text(json.dumps({'passed':True,'formal_calls':64,'retired_calls':retired['calls'],'updates':0,'local_CM_recomputed':True},indent=2))
 manifest={p.relative_to(R).as_posix():sha(p) for p in R.rglob('*') if p.is_file() and p.name!='LOCAL_MANIFEST.json'}
 (R/'LOCAL_MANIFEST.json').write_text(json.dumps(manifest,indent=2));print('sealed',len(manifest),'files')
if __name__=='__main__':main()
