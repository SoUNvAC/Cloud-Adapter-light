"""Independent scalar/count checks and concise Phase403 report; no GPU calls."""
import csv,json,hashlib,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase403/phase403_20261031'
def rows(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def close(a,b):assert abs(float(a)-float(b))<1e-8,(a,b)
def metrics(c):
 r=[sum(x) for x in c];p=[sum(c[i][j] for i in range(3)) for j in range(3)];u=[r[i]+p[i]-c[i][i] for i in range(3)];iou=[100*c[i][i]/u[i] if u[i] else None for i in range(3)]
 return {'IoU':iou,'mIoU':sum(v for v in iou if v is not None)/sum(v is not None for v in iou),'Shadow_IoU':iou[2],'Shadow_recall':100*c[2][2]/r[2] if r[2] else None,'nonShadow_FPR':100*(c[0][2]+c[1][2])/(r[0]+r[1]) if r[0]+r[1] else None}
def verify(row,c,prefix=''):
 for k,v in metrics(c).items():
  if k=='IoU':
   for a,b in zip(json.loads(row[prefix+k]),v):
    if b is None:assert a is None
    else:close(a,b)
  elif v is None:assert row[prefix+k] in ('','null')
  else:close(row[prefix+k],v)
def main():
 for n,h in js(R/'TRANSFER_ACCEPTED.json').items():assert sha(R/n)==h
 info=js(R/'RUN_INFO.json');lock=js(R/'INPUT_EXPECTED.json');assert info['passed'] and info['calls']==2268 and info['updates']==0 and info['seconds']<3600
 assert info['script_SHA']==lock['worker_SHA']==sha(ROOT/'src/tools/run_phase403.py')
 assert info['checkpoint_SHA']==lock['H_SHA'] and info['Source_SHA']==lock['Source_SHA']
 assert info['input_expected_SHA']==sha(R/'INPUT_EXPECTED.json') and info['ledger_SHA']==sha(R/'CALL_LEDGER.jsonl')
 ledger=[json.loads(x) for x in (R/'CALL_LEDGER.jsonl').read_text().splitlines()];assert len(ledger)==2268 and [x['calls'] for x in ledger]==list(range(1,2269)) and all(x['updates']==0 for x in ledger)
 s=rows(R/'summary.csv');g=rows(R/'per_group.csv');im=rows(R/'source_per_image.csv');assert len(s)==4 and len(g)==128 and len(im)==2140
 assert {x['case'] for x in s}=={'all_half','weak_2','weak_5','weak_2_5'}
 support={}
 for v in s:
  case=v['case']
  for domain,rr,n in [('target',g,32),('source',im,535)]:
   subset=[x for x in rr if x['case']==case];assert len(subset)==n and len({x['name'] for x in subset})==n
   c=[[0]*3 for _ in range(3)];refs={}
   for x in subset:
    cc=json.loads(x['confusion']);verify(x,cc);refs[x['name']]=[sum(y) for y in cc]
    for i in range(3):
     for j in range(3):c[i][j]+=cc[i][j]
   assert c==json.loads(v[domain+'_confusion']);verify(v,c,domain+'_')
   if domain in support:assert refs==support[domain]
   else:support[domain]=refs
   assert sum(x['current']['case']==case and x['current']['domain']==domain for x in ledger)==n
  aps=[float(x['AP']) for x in g if x['case']==case and x['AP']];assert len(aps)==7;close(sum(aps)/7,v['target_macro_AP'])
 prev=ROOT/'outputs/phase401/phase401_20261031';base=next(x for x in s if x['case']=='all_half');old=next(x for x in rows(prev/'summary.csv') if float(x['alpha'])==.5)
 for k in ['target_mIoU','source_mIoU','target_macro_AP','target_pooled_AP']:close(base[k],old[k])
 for file,new in [('per_group.csv',g),('source_per_image.csv',im)]:
  oldrows={x['name']:x for x in rows(prev/file) if float(x['alpha'])==.5}
  for x in new:
   if x['case']=='all_half':assert x['prediction_SHA']==oldrows[x['name']]['prediction_SHA'] and json.loads(x['confusion'])==json.loads(oldrows[x['name']]['confusion'])
 comparisons=[]
 for v in s:
  if v['case']=='all_half':continue
  for oldpoint in rows(prev/'summary.csv'):
   t=float(v['target_mIoU'])-float(oldpoint['target_mIoU']);src=float(v['source_mIoU'])-float(oldpoint['source_mIoU'])
   comparisons.append({'case':v['case'],'uniform_alpha':float(oldpoint['alpha']),'target_delta':t,'source_delta':src,'dominates_uniform':t>=0 and src>=0 and (t>0 or src>0),'dominated_by_uniform':t<=0 and src<=0 and (t<0 or src<0)})
 with (R/'uniform_comparisons.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(comparisons[0]));w.writeheader();w.writerows(comparisons)
 lines=['# Phase403：浅层减弱、深层保留诊断结果','','## 数据结果','','|配置|目标mIoU|源mIoU|目标Δ|源Δ|Shadow IoU|宏AP|','|---|---:|---:|---:|---:|---:|---:|']
 for v in s:lines.append('|'+v['case']+'|'+'|'.join(f'{x:.4f}' for x in [float(v['target_mIoU']),float(v['source_mIoU']),float(v['target_mIoU'])-float(base['target_mIoU']),float(v['source_mIoU'])-float(base['source_mIoU']),float(v['target_Shadow_IoU']),float(v['target_macro_AP'])])+'|')
 lines+=['','差值相对all_half。2268次前向、零训练；基线567张预测与Phase401相同。','',f"工作进程总耗时{info['seconds']:.1f}秒，包含加载、CPU统计和写盘；不是纯GPU时间。",'','## 解释边界','','单checkpoint、已暴露development与重复源val上的条件消融；不证明各层贡献可相加，不作为独立泛化或部署选择证据。逐组/逐图混淆与宏AP汇总独立复算通过；池化AP复用已验收实现，未传像元分数，未独立重算新点池化AP。']
 (R/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 (R/'INDEPENDENT_ACCEPTANCE.json').write_text(json.dumps({'passed':True,'checks':['input/code/checkpoint/transfer identity','2268 contiguous calls zero updates','every confusion and aggregate metrics','same support per sample','macro AP7','Phase401 baseline567 prediction SHA'],'pooled_AP_new_points_independent_pixel_recompute':False},indent=2))
 print('\n'.join(lines))
if __name__=='__main__':main()
