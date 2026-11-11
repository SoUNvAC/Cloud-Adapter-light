from pathlib import Path
T=Path(__file__).parent
def cv(s):return s.replace('479','480').replace('envelope','redistribution').replace('Envelope','Redistribution').replace('morphology','conserve')
def put(n,s):(T/n).write_bytes(s.encode('utf-8'))
for n in ['train_shadow479.py','infer_shadow479.py','verify_shadow479.py','ablate_shadow479.py','collect_shadow479.py','boundary_shadow479.py','verify_archive479.py','status_shadow479.py','run_shadow479.sh']:
 s=cv((T/n).read_text(encoding='utf-8'))
 if n=='train_shadow479.py':
  s=s.replace('loss=loss_ce+2*loss_keep+2*softloss+.01*delta.square().mean()', 'loss=loss_ce+2*loss_keep+2*softloss+.01*delta.square().mean()+.1*branch.redistribution.auxiliary(y)')
  s=s.replace('New multiscale dark-redistribution branch','New shadow subpatch redistribution branch')
  s=s.replace("position='SpatialCNN detail and context before all interactions'", "position='First VFM pyramid feature before decoder',shadow_fraction_auxiliary_weight=.1,patch_scale=4")
 put(n.replace('479','480'),s)
put('compare_shadow480.py',cv((T/'compare_shadow479.py').read_text(encoding='utf-8')))
put('finish_shadow480.sh',cv((T/'finish_shadow479.sh').read_text(encoding='utf-8')))
s=cv((T/'diagnose_envelope479.py').read_text(encoding='utf-8'))
s=s.replace("values=[net.branch.redistribution.relative_change.item()]", "values=[net.branch.redistribution.relative_change.item()];assert net.branch.redistribution.mean_residual.item()<1e-5")
s=s.replace("mean_relative_detail_rms", "mean_relative_detail_rms").replace('ENVELOPE_DIAG','REDISTRIBUTION_DIAG')
put('diagnose_redistribution480.py',s)
