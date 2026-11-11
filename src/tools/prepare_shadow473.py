from pathlib import Path
T=Path(__file__).parent
names=['train_shadow472.py','infer_shadow472.py','verify_shadow472.py','boundary_shadow472.py','ablate_shadow472.py','ablate_appearance472.py','collect_shadow472.py','compare_shadow472.py','monitor_shadow472.py','run_shadow472.sh','audit_shadow472.py','accept_shadow472.py']
for name in names:
 s=(T/name).read_text(encoding='utf-8').replace('472','473')
 if name=='train_shadow472.py':
  s=s.replace('loss=loss_ce+2*loss_keep+2*softloss+.01*delta.square().mean()',"loss=loss_ce+2*loss_keep+2*softloss+.01*delta.square().mean()\n  if ARM=='B':loss=loss+.1*branch.contrast.auxiliary(y)")
  s=s.replace('auxiliary_loss=False','auxiliary_loss=True,auxiliary_weight=.1')
 if name=='compare_shadow472.py':
  s=s.replace("d=R/arm/'network_run'","d=(Path('/home/scv/shared/phase472_20261107') if arm=='A' else R)/arm/'network_run'")
  s=s.replace("(R/a/'network_run/HISTORY.json')","((Path('/home/scv/shared/phase472_20261107') if a=='A' else R)/a/'network_run/HISTORY.json')")
 if name=='accept_shadow472.py':
  s=s.replace("R/arm", "(Path('/home/scv/shared/phase472_20261107') if arm=='A' else R)/arm")
 (T/name.replace('472','473')).write_bytes(s.encode('utf-8'))
