"""Local receipt verification and concise history entry; never trains."""
from pathlib import Path
import csv,json,hashlib,sys
ROOT=Path(__file__).resolve().parents[2]
def js(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(phase):
 r=ROOT/f'outputs/phase{phase}/phase{phase}_20261101'
 for n,h in js(r/'TRANSFER_MANIFEST.json').items():assert sha(r/n)==h,n
 receipt=js(r/'REPORT_COMPLETE.json');assert receipt['passed']
 for n,h in receipt['files'].items():assert sha(r/n)==h,n
 for n,h in js(r/'REPORT_CODE_LOCK.json').items():assert sha(r/n)==h,n
 a=js(r/'INDEPENDENT_ACCEPTANCE.json');complete=js(r/'COMPLETE.json');assert a['passed'] and complete['passed']
 assert a['calls']==complete['new_calls'] and a['updates']==complete['new_updates']
 with (r/'TARGET_SUMMARY.csv').open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
 for row in rows:
  cm=json.loads(row['confusion']);ious=[]
  for i in range(3):ious.append(100*cm[i][i]/(sum(cm[i])+sum(c[i] for c in cm)-cm[i][i]))
  assert abs(sum(ious)/3-float(row['mIoU']))<1e-9
 endpoint=[v for v in rows if int(v['step'])==1000 and not v['mode'].endswith('_full')]
 assert len(endpoint)==(6 if phase==412 else 8)
 baseline_rows=list(csv.DictReader((ROOT/'outputs/phase411/phase411_20261101/TARGET_SUMMARY.csv').open()))
 baseline=next(float(v['mIoU']) for v in baseline_rows if v['mode']=='layerwise' and v['step']=='1000')
 assert abs(js(r/'PROMOTION.json')['baseline_1000']-baseline)<1e-9
 title=f"## Phase{phase}："+('三种动态MsRE机制' if phase==412 else '云影双算子、位移关联与共享照明场')+'（2026-11-01）'
 entry=title+'\n\n- 153 fit / 32 development，seed40901；以Phase411分层MsRE为底座，H关闭。每组1000步，每500步验证，仅第1000步筛选；源域不参与选择。\n'
 for v in endpoint:entry+=f"- {v['mode']}：mIoU {float(v['mIoU']):.4f}，影IoU {float(v['Shadow_IoU']):.4f}，影宏AP {float(v['macro_AP']):.4f}。\n"
 entry+='- 晋级4000步：'+(', '.join(complete['selected']) or '无，未同时超过对应所有对照及Phase411至少0.30点')+'。单seed短程筛选，不代表最终上限或稳定性。\n'
 retired_calls=sum(v['calls'] for v in a['retired_attempts']);retired_updates=sum(v['completed_update_trace_rows'] for v in a['retired_attempts'])
 entry+=f"- 有效运行{a['calls']}次前向、{a['updates']}次更新；另有保留的修正尝试{retired_calls}次已记录前向、{retired_updates}条已完成更新记录。详见验收中的原因和远端归档。\n"
 entry+='- 数据/增强/学习率配对、初始权重、冻结权重、调用、逐组混淆矩阵与精确AP、checkpoint身份及晋级规则验收通过。\n'
 if phase==413:entry+='- 所有对照同样获得原始RGB和语义辅助损失；只将组内差值用于机制判断。物理算子、软参考及位移均是代理机制，未反演真实物理参数。\n'
 entry+=f'- 交付outputs/phase{phase}/phase{phase}_20261101/RESULT.md及指标、代码锁、调用和验收记录。\n\n'
 log=ROOT/'src/EXPERIMENT_LOG.md';raw=log.read_bytes();marker='## 后续维护规则'.encode();assert raw.count(marker)==1
 if title.encode() not in raw:log.write_bytes(raw.replace(marker,entry.replace('\n','\r\n').encode()+marker))
 p=ROOT/f'src/research_plans/phase{phase}.md';s=p.read_text(encoding='utf-8')
 status=f'完成：2026-11-01；有效运行{a["calls"]}次前向、{a["updates"]}次更新，验收通过。结果见outputs/phase{phase}/phase{phase}_20261101/RESULT.md。'
 if status not in s:p.write_text(s+'\n'+status+'\n',encoding='utf-8')
 (r/'README.md').write_bytes((r/'RESULT.md').read_bytes())
 (r/'DELIVERY_COMPLETE.json').write_text(json.dumps({'passed':True,'phase':phase,'calls':a['calls'],'updates':a['updates'],'retired_calls':retired_calls,'selected':complete['selected'],'source_selection_gate':False,'local_mIoU_recomputed':True},indent=2))
 manifest={p.relative_to(r).as_posix():sha(p) for p in r.rglob('*') if p.is_file() and p.name!='LOCAL_MANIFEST.json'}
 (r/'LOCAL_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
 print('sealed',phase,len(manifest),'files')
if __name__=='__main__':main(int(sys.argv[1]))
