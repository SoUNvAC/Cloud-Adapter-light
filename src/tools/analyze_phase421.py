"""Scene/class attribution from all 32 saved confusion matrices; no new inference."""
from pathlib import Path
import json,csv
R=Path(__file__).resolve().parents[2]/'outputs/phase421/phase421_20261101'
def js(p):return json.loads(p.read_text(encoding='utf-8'))
def miou(cm):return sum(100*cm[i][i]/(sum(cm[i])+sum(r[i] for r in cm)-cm[i][i]) for i in range(3))/3
def main():
 models={mode:js(R/mode/'evaluations/M/step_1000/METRICS.json') for mode in ['baseline','neighborhood','material']}
 base=models['baseline'];basegroups={g['product']:g for g in base['per_group']};rows=[]
 for mode in ['neighborhood','material']:
  for g in models[mode]['per_group']:
   cm=g['confusion'];ref=basegroups[g['product']]['confusion']
   replaced=[[base['confusion'][i][j]-ref[i][j]+cm[i][j] for j in range(3)] for i in range(3)]
   row={'mode':mode,'product':g['product'],'single_scene_substitution_delta_pp':miou(replaced)-base['mIoU']}
   for i,gt in enumerate(['surface','cloud','shadow']):
    for j,pred in enumerate(['surface','cloud','shadow']):
     if i!=j:row[gt+'_to_'+pred+'_delta']=cm[i][j]-ref[i][j]
   rows.append(row)
 with (R/'SCENE_ERROR_CHANGES.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 lines=['# Phase421 结果解释','','所有32组场景均保留，未调整标签。单场景替换只用于定位错误变化，不能把替换后的数值当成模型成绩。','']
 for mode in ['neighborhood','material']:
  m=models[mode];lines.append(f"## {mode}")
  lines.append(f"相对基线mIoU {m['mIoU']-base['mIoU']:+.4f}个百分点。各类IoU变化对总mIoU的贡献（除以3）：")
  for i,n in enumerate(['地表','云','阴影']):lines.append(f"- {n}：{(m['IoU'][i]-base['IoU'][i])/3:+.4f}点。")
  ranked=sorted([r for r in rows if r['mode']==mode],key=lambda r:r['single_scene_substitution_delta_pp'])
  lines.append('按单场景替换影响排序，最差两场景与最好两场景：')
  for r in ranked[:2]+ranked[-2:]:lines.append(f"- {r['product']}：{r['single_scene_substitution_delta_pp']:+.4f}点。")
  lines.append('')
 margin=models['material']['mIoU']-max(models[m]['mIoU'] for m in ['baseline','neighborhood'])
 lines.append(f'材质版超过最强对照 {margin:+.4f}点。'+('本轮支持安排独立seed复验，尚不能认定稳定。' if margin>0 else '本轮不支持扩大训练；只否定本次局部参考实现的短程收益，不否定所有材质关系建模。'))
 (R/'INTERPRETATION.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print('scene and class attribution saved')
if __name__=='__main__':main()
