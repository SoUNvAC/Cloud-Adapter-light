from pathlib import Path
T=Path(__file__).parent
for name in ['shadow_conditional474.py','train_shadow474.py','infer_shadow474.py','verify_shadow474.py','boundary_shadow474.py','collect_shadow474.py','accept_shadow474.py','ablate_shadow474.py','ablate_spatial474.py','verify_archive474.py']:
 s=(T/name).read_text().replace('474','475')
 if name=='train_shadow474.py':
  s=s.replace('torch.manual_seed(47501)','torch.manual_seed(47401)').replace("conditional=ARM=='B'",'conditional=False')
  s=s.replace('weights=torch.ones_like(y,dtype=torch.float);weights[hard]=4.;weights[(y==2)&(old==2)]=1.5',"weights=torch.ones_like(y,dtype=torch.float)\n  if ARM=='A':weights[hard]=4.")
  s=s.replace("new_module=dict(conditional=False", "loss_variant='hard4_correct_shadow1' if ARM=='A' else 'uniform1',new_module=dict(conditional=False")
 (T/name.replace('474','475')).write_bytes(s.encode())
