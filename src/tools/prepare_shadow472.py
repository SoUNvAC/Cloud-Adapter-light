from pathlib import Path
T=Path(__file__).parent
def put(n,s):(T/n).write_bytes(s.encode('utf-8'))
def copy(n):return (T/n).read_text(encoding='utf-8').replace('471','472')
s=copy('train_shadow471.py')
s=s.replace('for n,p in student.named_parameters():p.requires_grad_(False)',"for n,p in student.named_parameters():p.requires_grad_(n.startswith(('decode_head.','backbone.refinement.projector.')))")
a=s.index(' from shadow_transport472 import');b=s.index(' frozen_hash=',a)
s=s[:a]+''' if ARM=='B':
  from shadow_contrast472 import ContrastProjector,install
  with torch.random.fork_rng(devices=[0]):
   torch.manual_seed(47201);branch.contrast=ContrastProjector().cuda()
  install(student.backbone,branch.contrast)
 frozen_items=lambda:[(n,t) for n,t in student.state_dict().items() if not n.startswith(('decode_head.','backbone.refinement.projector.'))]
'''+s[b:]
s=s.replace('if step>0 and metrics', 'if step>=2000 and metrics').replace('if protected and metrics', 'if (step==0 or step>=2000) and protected and metrics')
s=s.replace("row=dict(step=step,**metrics,", "row=dict(step=step,contrast_strength=float(branch.contrast.strength) if ARM=='B' else 0.,**metrics,")
s=s.replace("  y=y.cuda(non_blocking=True);", "  if ARM=='B':branch.contrast.strength.fill_(.25*min(step/2000,1))\n  y=y.cuda(non_blocking=True);")
s=s.replace('  loss=loss+.1*branch.transport.auxiliary(y,reference)\n','')
s=s.replace('   assert not active and sum(changed.values())==0,changed',"   assert changed['backbone.refinement.']>0 and changed['decode_head.']>0,changed\n   assert sum(v for k,v in changed.items() if k not in ['backbone.refinement.','decode_head.'])==0")
s=s.replace("   assert any(not torch.equal(initial_branch[n],p.detach().cpu()) for n,p in branch.named_parameters() if n.startswith('transport.'))", "   if ARM=='B':assert all(not torch.equal(initial_branch[n],p.detach().cpu()) for n,p in branch.named_parameters() if n.startswith('contrast.'))")
s=s.replace("'transport.'","'contrast.'")
s=s.replace('Only new contour transport layer trainable; all original parameters and BN fixed','Original MsRE projector and decoder trainable; new B contrast selector trainable; all other weights and BN fixed')
s=s.replace('Best trained shadow plus best nonshadow-protected relative to450; origin step0 is fallback','Select trained checkpoints from step2000 onward at full B strength0.25; origin step0 is fallback')
s=s.replace("dict(normal=ARM=='B',max_flow_grid=2,zero_init=True,distance_aux_weight=.1,position='First decoder pyramid feature, quarter resolution')","dict(enabled=ARM=='B',strength=.25,ramp_steps=2000,zero_strength_origin=True,auxiliary_loss=False,position='Original MsRE projector input, all4 stages')")
s=s.replace('shadow_transport472.py','shadow_contrast472.py')
put('train_shadow472.py',s)
s=copy('infer_shadow471.py').replace('shadow_transport472','shadow_contrast472').replace('ContourTransport','ContrastProjector').replace('transport.','contrast.').replace('branch.transport','branch.contrast')
s=s.replace("ContrastProjector(normal=bool(branch_state['contrast.normal_mode']))",'ContrastProjector()')
put('infer_shadow472.py',s)
for n in ['verify_shadow471.py','boundary_shadow471.py','monitor_shadow471.py']:
 put(n.replace('471','472'),copy(n))
put('ablate_shadow472.py',copy('ablate_shadow471.py').replace('branch.transport','branch.contrast'))
put('ablate_appearance472.py',copy('ablate_normal471.py').replace('branch.transport','branch.contrast').replace('normal_enabled','appearance_enabled').replace('normal_off','appearance_off').replace('NORMAL_OFF','APPEARANCE_OFF'))
put('collect_shadow472.py',copy('collect_shadow471.py').replace('normal_off','appearance_off'))
put('compare_shadow472.py',copy('compare_shadow471.py').replace('B_transport_on_minus_off','B_contrast_on_minus_off').replace('B_normal_on_minus_off','B_appearance_on_minus_off').replace('NORMAL_OFF','APPEARANCE_OFF'))
put('run_shadow472.sh',copy('run_shadow471.sh').replace('ablate_normal472','ablate_appearance472'))
