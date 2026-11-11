from pathlib import Path
T=Path(__file__).parent
def put(n,s):(T/n).write_bytes(s.encode())
s=(T/'train_shadow472.py').read_text().replace('472','476')
s=s.replace("('decode_head.','backbone.refinement.projector.')","('decode_head.',)")
a=s.index(" if ARM=='B':\n  from shadow_contrast476");b=s.index(' frozen_items=',a)
s=s[:a]+''' from shadow_material476 import MaterialConvolution,install
 with torch.random.fork_rng(devices=[0]):
  torch.manual_seed(47601);branch.material=MaterialConvolution(chroma=ARM=='B').cuda()
 install(student.backbone,branch.material)
'''+s[b:]
s=s.replace('lr=1e-4,peak=1e-4','lr=1e-3,peak=1e-3').replace('branch_lr=1e-4','branch_lr=1e-3')
s=s.replace('if step>=2000 and metrics','if step>0 and metrics').replace('if (step==0 or step>=2000) and protected','if protected')
s=s.replace("contrast_strength=float(branch.contrast.strength) if ARM=='B' else 0.,",'')
s=s.replace("  if ARM=='B':branch.contrast.strength.fill_(.25*min(step/2000,1))\n",'')
s=s.replace('weights=torch.ones_like(y,dtype=torch.float);weights[hard]=4.;weights[(y==2)&(old==2)]=1.5','weights=torch.ones_like(y,dtype=torch.float)')
s=s.replace("changed['backbone.refinement.']>0 and ",'').replace("['backbone.refinement.','decode_head.']","['decode_head.']")
s=s.replace("if ARM=='B':assert all",'assert all').replace('contrast.','material.')
s=s.replace('Original MsRE projector and decoder trainable; new B contrast selector trainable; all other weights and BN fixed','New spatial material convolution and original decoder trainable; all other weights and BN fixed')
s=s.replace('Select trained checkpoints from step2000 onward at full B strength0.25; origin step0 is fallback','Best trained shadow and protected checkpoint; origin step0 fallback')
s=s.replace("dict(enabled=ARM=='B',strength=.25,ramp_steps=2000,zero_strength_origin=True,auxiliary_loss=False,position='Original MsRE projector input, all4 stages')","dict(chroma=ARM=='B',zero_difference_kernel=True,uniform_pixel_CE=True,position='SpatialCNN detail and context before all interactions')")
s=s.replace('shadow_contrast476.py','shadow_material476.py');put('train_shadow476.py',s)
s=(T/'infer_shadow472.py').read_text().replace('472','476').replace('contrast.','material.').replace('branch.contrast','branch.material').replace('shadow_contrast476','shadow_material476').replace('ContrastProjector()','MaterialConvolution(chroma=bool(branch_state[\'material.chroma_mode\']))').replace('ContrastProjector,install','MaterialConvolution,install')
put('infer_shadow476.py',s)
for n in ['verify_shadow472.py','boundary_shadow472.py','collect_shadow472.py','compare_shadow472.py']:
 s=(T/n).read_text().replace('472','476');put(n.replace('472','476'),s)
s=(T/'ablate_shadow472.py').read_text().replace('472','476').replace('branch.contrast','branch.material');put('ablate_shadow476.py',s)
s=(T/'ablate_appearance472.py').read_text().replace('472','476').replace('branch.contrast','branch.material').replace('appearance_enabled','chroma_enabled');put('ablate_appearance476.py',s)
put('run_shadow476.sh',(T/'run_shadow472.sh').read_text().replace('472','476'))
