from pathlib import Path
T=Path(__file__).parent
def cv(s):return s.replace('479','480').replace('envelope','redistribution').replace('Envelope','Redistribution').replace('morphology','conserve')
s=cv((T/'accept_shadow479.py').read_text(encoding='utf-8')).replace('len(new_changed)==6','len(new_changed)==13').replace("==4992","==15265")
(T/'accept_shadow480.py').write_bytes(s.encode())
s=(T/'diagnose_redistribution480.py').read_text(encoding='utf-8')
s=s.replace("relative_input_rms=values)","relative_input_rms=values,patch_mean_residual_max=net.branch.redistribution.mean_residual.item())")
s=s.replace("(R/'REDISTRIBUTION_DIAG.json')","result['maximum_patch_mean_residual']=max(v['patch_mean_residual_max'] for v in per)\n(R/'REDISTRIBUTION_DIAG.json')")
(T/'diagnose_redistribution480.py').write_bytes(s.encode())
