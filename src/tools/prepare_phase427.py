"""Generate Phase427's isolated backbone, training controller and delivery tools."""
from pathlib import Path
import ast
T=Path(__file__).parent;ROOT=T.parents[1]
template=ROOT/'outputs/phase426/phase426_20261101/baseline/code/phase400_model.py'
s=template.read_text().replace('from phase426_competition import CompetitiveMsRE as SparseMsREResidual',
    'from phase427_fused import SpatialMsRE as SparseMsREResidual, attach\nimport os')
s=s.replace('  self.set_target_enabled(self.target_enabled)',
'''  self.mode=os.environ['PHASE427_MODE']
  msre=self.target_msre
  del self.target_msre
  attach(self.cloud_adapter,msre,self.mode)
  self.set_target_enabled(self.target_enabled)''',1)
start=s.index(' def forward_features(')
s=s[:start]+''' def __getattr__(self,name):
  if name=='target_msre':
   modules=self.__dict__.get('_modules',{})
   if 'cloud_adapter' in modules and hasattr(modules['cloud_adapter'],'msre'):
    return modules['cloud_adapter'].msre
  return super().__getattr__(name)
 def forward_features(self,x,masks=None):
  b,_,height,width=x.shape;h=height//self.patch_size;w=width//self.patch_size
  cache,detail=self.cloud_adapter.prepare(x)
  x=self.prepare_tokens_with_masks(x,masks);outputs=[];slot=0
  for index,block in enumerate(self.blocks):
   if index not in self._active_block_index_set:continue
   x=block(x)
   if index in self.adapter_index:
    x,cache,detail=self.cloud_adapter.interact(x,slot,index,h,w,cache,detail);slot+=1
   if index in self.out_indices:
    outputs.append(x[:,1:].transpose(1,2).reshape(b,-1,h,w).contiguous())
  return outputs,cache
 def train(self,mode=True):
  nn.Module.train(self,False)
  if not mode:return self
  active=['cloud_adapter.msre.']
  if self.mode!='baseline':active.append('cloud_adapter.net.')
  if self.mode=='integrated':active.append('cloud_adapter.detail.')
  for name,p in self.named_parameters():p.requires_grad=any(name.startswith(k) for k in active)
  self.cloud_adapter.msre.train()
  if self.mode!='baseline':self.cloud_adapter.net.train()
  if self.mode=='integrated':self.cloud_adapter.detail.train()
  return self
'''
(T/'phase427_backbone.py').write_text(s,encoding='utf-8')
s=(T/'controller_phase426.py').read_text().replace('426','427').replace('phase427_competition','phase427_fused')
start=s.index("  if n=='phase400_model.py':")
end=s.index("  if n=='phase400_runtime.py':",start)
s=s[:start]+"  if n=='phase400_model.py':t=(R/'phase427_backbone.py').read_text()\n"+s[end:]
s=s.replace("('router.',)","('detail.',)")
old="   t=t.replace(\"cfg.model.backbone.type='Phase400TargetMsRECloudAdapter'\", \"cfg.model.type='Phase427Segmentor';cfg.model.backbone.type='Phase400TargetMsRECloudAdapter'\")"
new="   t=t.replace(\"cfg.model.backbone.type='Phase400TargetMsRECloudAdapter'\", \"cfg.model.backbone.type='Phase400TargetMsRECloudAdapter';cfg.model.backbone.cloud_adapter_config.type='Phase427CloudAdapter'\")"
assert old in s;s=s.replace(old,new)
needle="  if n=='phase400_common.py':"
extra='''   original="permitted=('backbone.target_msre.','backbone.target_head_delta.')"
   allowed=['backbone.cloud_adapter.msre.','backbone.target_head_delta.']
   if mode!='baseline':allowed.append('backbone.cloud_adapter.net.')
   if mode=='integrated':allowed.append('backbone.cloud_adapter.detail.')
   assert original in t;t=t.replace(original,'permitted='+repr(tuple(allowed)))
   t=t.replace("'head_initial_SHA':", "'source_interaction_initial_SHA':{n:tensor_sha(v) for n,v in model.state_dict().items() if n.startswith('backbone.cloud_adapter.net.')},'head_initial_SHA':")
'''
s=s.replace(needle,extra+needle)
s=s.replace("reports[mode]=run(mode,1000,tests[mode]['parameters'])","reports[mode]=run(mode,1000,tests[mode]['parameters'])")
(T/'controller_phase427.py').write_text(s,encoding='utf-8')
s=(T/'launch_phase426.py').read_text().replace('426','427').replace('phase427_competition','phase427_fused')
s=s.replace("'phase427_fused.py']","'phase427_fused.py','phase427_backbone.py']")
(T/'launch_phase427.py').write_text(s,encoding='utf-8')
s=(T/'collect_phase426.py').read_text().replace('426','427');(T/'collect_phase427.py').write_text(s,encoding='utf-8')
s=(T/'status_phase426.py').read_text().replace('426','427').replace("['baseline','competitive']","['baseline','unfrozen','integrated']")
(T/'status_phase427.py').write_text(s,encoding='utf-8')
s=(T/'report_phase426.py').read_text()
s=s.replace("assert scope['frozen_SHA']==refscope['frozen_SHA'] and scope['head_initial_SHA']==refscope['head_initial_SHA']", "assert all(refscope['frozen_SHA'][k]==v for k,v in scope['frozen_SHA'].items())\n  assert all(refscope['frozen_SHA'][k]==v for k,v in scope['source_interaction_initial_SHA'].items())\n  assert scope['head_initial_SHA']==refscope['head_initial_SHA']")
s=s.replace("k.startswith('backbone.target_msre.')","k.startswith(('backbone.cloud_adapter.msre.','backbone.cloud_adapter.detail.','backbone.cloud_adapter.net.'))")
start=s.index(" a,b=endpoint['competitive']")
end=s.index(" save('REPORT_COMPLETE.json'",start)
s=s[:start]+''' comparisons=[]
 for mode,control in [('unfrozen','baseline'),('integrated','baseline'),('integrated','unfrozen')]:
  a,b=endpoint[mode],endpoint[control]
  comparisons.append({'mode':mode,'control':control,'delta_mIoU':a['mIoU']-b['mIoU'],'delta_shadow_IoU':a['Shadow_IoU']-b['Shadow_IoU'],'delta_macro_AP':a['macro_AP']-b['macro_AP']})
 accepted={'passed':True,'checks':checks,'calls':sum(c['calls'] for c in costs),'updates':sum(c['steps'] for c in costs),'CPU_report_seconds':time.monotonic()-start,'scope':'DINO/CNN/head frozen; attention trainable in unfrozen/integrated'}
 save('INDEPENDENT_ACCEPTANCE.json',accepted);writecsv(R/'COMPARISONS.csv',comparisons)
 save('MECHANISM_SCREEN.json',{'comparisons':comparisons,'scope':'single seed fixed endpoint full integrated package'})
 lines=['# Phase427 结果','','153 fit / 完整32 development，seed40901；固定1000步为主。','', '|方案|步数|mIoU|地表IoU|云IoU|阴影IoU|阴影宏AP|','|---|---:|---:|---:|---:|---:|---:|']
 for row in rows:
  m=js(R/row['mode']/f"evaluations/M/step_{row['step']}/METRICS.json")
  lines.append(f"|{row['mode']}|{row['step']}|{row['mIoU']:.4f}|{m['IoU'][0]:.4f}|{m['IoU'][1]:.4f}|{row['Shadow_IoU']:.4f}|{row['macro_AP']:.4f}|")
 for c in comparisons:lines.append(f"\\n{c['mode']} 相对 {c['control']}：mIoU {c['delta_mIoU']:+.4f}点；阴影IoU {c['delta_shadow_IoU']:+.4f}点；宏AP {c['delta_macro_AP']:+.4f}点。")
 lines+=['',f"共{accepted['calls']}次前向、{accepted['updates']}次更新，GPU阶段累计{sum(c['seconds'] for c in costs)/60:.2f}分钟。",'', '|方案|可训练参数|平均训练步秒数|','|---|---:|---:|']
 for c in costs:lines.append(f"|{c['mode']}|{c['parameters']}|{c['mean_step_seconds']:.4f}|")
 lines+=['','MsRE已归属Cloud-Adapter，融合组将1/4尺度局部细节送入多尺度残差，并把更新语义反馈空间缓存。原分割损失，无辅助监督。','unfrozen/integrated允许原交互层训练；DINOv2、CNN、分割头冻结。共同初始MsRE、Source交互权重、输入/标签/增强/LR配对与完整预测/AP复算通过。单seed开发集探索，未触及确认集。']
 (R/'RESULT.md').write_text('\\n'.join(lines)+'\\n',encoding='utf-8')
''' +s[end:]
(T/'report_phase427.py').write_text(s,encoding='utf-8')
s=(T/'launch_report_phase426.py').read_text().replace('426','427')
(T/'launch_report_phase427.py').write_text(s,encoding='utf-8')
for p in T.glob('*phase427*.py'):ast.parse(p.read_text(encoding='utf-8'))
print('Phase427 scripts parsed')
