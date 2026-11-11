from pathlib import Path
T=Path(__file__).parent
s=(T/'train_shadow476.py').read_text().replace('476','477')
a=s.index(' from shadow_material477 import');b=s.index(' frozen_items=',a);s=s[:a]+s[b:]
s=s.replace("assert ARM in ['A','B']","assert ARM=='A'")
s=s.replace("new_module=dict(chroma=ARM=='B',zero_difference_kernel=True,uniform_pixel_CE=True,position='SpatialCNN detail and context before all interactions')","new_module=None,uniform_pixel_CE=True")
s=s.replace(",Path(__file__).with_name('shadow_material477.py')",'')
s=s.replace('branch_lr=1e-3','branch_lr=0.0').replace('New spatial material convolution and original decoder trainable; all other weights and BN fixed','Only original decoder trainable; original backbone,450 branch and BN fixed')
s=s.replace("branch_updated=ARM=='B'","branch_updated=False")
s=s.replace("   assert all(not torch.equal(initial_branch[n],p.detach().cpu()) for n,p in branch.named_parameters() if n.startswith('material.'))\n",'')
(T/'train_shadow477.py').write_bytes(s.encode())
s=(T/'infer_shadow476.py').read_text().replace('476','477');a=s.index("        if any(k.startswith('material.')");b=s.index('        self.branch.load_state_dict',a);s=s[:a]+s[b:];(T/'infer_shadow477.py').write_bytes(s.encode())
for n in ['verify_shadow476.py','boundary_shadow476.py','collect_shadow476.py','verify_archive476.py']:
 s=(T/n).read_text().replace('476','477')
 if n=='verify_shadow476.py':s=s.replace("choices={'best_shadow':done['best_shadow']['step']}","choices={'best_shadow':done['best_shadow']['step'],'matched_1000':1000,'matched_3000':3000,'last_6000':6000}")
 (T/n.replace('476','477')).write_bytes(s.encode())
