from pathlib import Path
T=Path(__file__).parent
def put(n,s):(T/n).write_bytes(s.encode('utf-8'))
def rename(s):
 for prefix in ['phase','shadow','redistribution','effect','region','archive','all']:s=s.replace(prefix+'493',prefix+'495')
 return s.replace("'493'","'495'").replace('493{','495{').replace('493A_','495A_').replace('493B_','495B_').replace("'493A'","'495A'").replace("'493B'","'495B'")
for stem in ['train_shadow','infer_shadow','verify_shadow','ablate_shadow','boundary_shadow','boundary_effect','boundary_region','ablate_region','verify_archive','collect_all','finish_shadow','status_shadow','cost_shadow','compare_shadow']:
 s=rename((T/f'{stem}493.py').read_text(encoding='utf-8'))
 if stem=='train_shadow':
  s=s.replace("ordinal=ARM=='A'","ordinal=False,feedback_slot=0 if ARM=='A' else 3")
  s=s.replace("position='First VFM pyramid feature before decoder'","position='Retained first-pyramid path plus post-MsRE token feedback'")
  s=s.replace("save('UPDATE_CHECK.json',dict(step=step,","save('UPDATE_CHECK.json',dict(feedback_relative=float(branch.redistribution.feedback_relative),feedback_calls=branch.redistribution.feedback_calls,step=step,")
 if stem=='infer_shadow':s=s.replace("ordinal=bool(branch_state.get('redistribution.ordinal_mode',True))","ordinal=False,feedback_slot=int(branch_state['redistribution.feedback_slot'])")
 if stem=='ablate_region':s=s.replace('"ordinal_enabled",False','"feedback_enabled",False')
 if stem in ['compare_shadow','cost_shadow']:
  s=s.replace('phase488_20261108','phase493_20261109').replace('488B','493B').replace('infer_shadow488','infer_shadow493').replace("['488']","['493']")
 if stem=='compare_shadow':s=s.replace('default_rng(49302)','default_rng(49502)')
 put(f'{stem}495.py',s)
put('run_shadow495.sh',rename((T/'run_shadow493.sh').read_text()))
put('collect_metadata495.py',rename((T/'collect_metadata493.py').read_text()))
s=(T/'shadow_redistribution493.py').read_text()
s=s.replace('ordinal=True):','ordinal=False,feedback_slot=0):')
s=s.replace('self.ordinal_enabled=True','''self.ordinal_enabled=True
  self.register_buffer('feedback_slot',torch.tensor(feedback_slot,dtype=torch.int64))
  self.feedback_inject=nn.Conv2d(16,384,1,bias=False);nn.init.zeros_(self.feedback_inject.weight)
  self.feedback_enabled=True;self.ordinal_field=None;self.feedback_relative=0.;self.feedback_calls=0''')
s=s.replace('self.ordinal(self.ordinal_evidence(self.rgb))','self.ordinal_field')
pos=s.index(' def texture_evidence(')
s=s[:pos]+''' def prepare_rgb(self,rgb):
  self.rgb=rgb;self.feedback_calls=0;self.feedback_relative=0.
  self.ordinal_field=self.ordinal(self.ordinal_evidence(rgb)) if self.enabled and self.evidence_enabled and self.ordinal_enabled else None
 def feedback(self,out,slot,h,w):
  if not self.enabled or not self.feedback_enabled or self.ordinal_field is None or slot!=int(self.feedback_slot):return out
  delta=self.feedback_inject(F.adaptive_avg_pool2d(self.ordinal_field,(h,w))).flatten(2).transpose(1,2)
  # Match token units without differentiating through the scale estimate.
  scale=out.detach().float().square().mean(-1,keepdim=True).sqrt().clamp_min(1e-4)
  update=delta*scale
  self.feedback_relative=(update.detach().square().mean()/(out.detach().square().mean()+1e-12)).sqrt()
  self.feedback_calls+=1
  return out+update
''' +s[pos:]
s=s.replace('module.rgb=((x*std+mean)/255).clamp(0,1)','module.prepare_rgb(((x*std+mean)/255).clamp(0,1))')
s=s.replace('backbone.register_forward_pre_hook(rgb_hook)','''backbone.register_forward_pre_hook(rgb_hook)
 backbone.refinement.register_forward_hook(lambda m,args,out:module.feedback(out,args[2],args[3],args[4]))''')
put('shadow_redistribution495.py',s)
# Frozen-parameter and real-checkpoint acceptance, adapted from the tested single-seed pipeline.
s=rename((T/'accept_shadow493.py').read_text()).replace('phase488_20261108','phase493_20261109').replace('to_488B','to_493B')
s=s.replace("if n!='redistribution.ordinal_mode'","if n!='redistribution.feedback_slot'")
s=s.replace("lock['new_module']['ordinal']==(arm=='A')","not lock['new_module']['ordinal'] and lock['new_module']['feedback_slot']==(0 if arm=='A' else 3)")
s=s.replace("==5","==2").replace("len(new_changed)==21","len(new_changed)==22")
s=s.replace("bool(origin['state_dict']['redistribution.ordinal_mode'])==(arm=='A')","not bool(origin['state_dict']['redistribution.ordinal_mode']) and int(origin['state_dict']['redistribution.feedback_slot'])==(0 if arm=='A' else 3)")
s=s.replace("bool(s['state_dict']['redistribution.ordinal_mode'])==(arm=='A')","not bool(s['state_dict']['redistribution.ordinal_mode']) and int(s['state_dict']['redistribution.feedback_slot'])==(0 if arm=='A' else 3)")
s=s.replace("u['step']==6000 and all(u['new_changed'].values())","u['step']==6000 and all(u['new_changed'].values()) and u['feedback_calls']==1 and u['feedback_relative']>0")
put('accept_shadow495.py',s)
