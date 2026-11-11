"""Generate two-domain fixed tables and independent count/bootstrap validators."""
from pathlib import Path
import ast
T=Path(__file__).resolve().parent
def names(t):return t.replace("'R'","'W1'").replace("'Q'","'W10'").replace('R_prediction_SHA','W1_prediction_SHA').replace('Q_prediction_SHA','W10_prediction_SHA')
def comparisons(t):
 t=t.replace("for other in ['W10','D','H']:","for candidatearm,other in [('W1','D'),('W10','D'),('W1','H'),('W10','H')]:")
 a=t.index(" for candidatearm,other in");b=t.index(" finite=[]",a) if " finite=[]" in t[a:] else t.index(" joint=collections",a)
 sub=t[a:b].replace("'W1'",'candidatearm').replace("'R-'+other","candidatearm+'-'+other").replace("'R_alpha':x","'candidate_arm':candidatearm,'candidate_alpha':x").replace("'R_alpha':'/'.join(str(v['R_alpha']) for v in individual)","'candidate_arm':candidatearm,'candidate_alpha':'/'.join(str(v['candidate_alpha']) for v in individual)")
 sub=sub.replace("for candidatearm,other in [(candidatearm,'D'),('W10','D'),(candidatearm,'H'),('W10','H')]:","for candidatearm,other in [('W1','D'),('W10','D'),('W1','H'),('W10','H')]:")
 return t[:a]+sub+t[b:]
for domain,prior in [('sparcs',226),('l8',225)]:
 t=names((T/f'analyze_phase{prior}.py').read_text()).replace(f'phase{prior}/phase{prior}_'+('20261025' if prior==226 else '20261024'),'phase227/phase227_20261025').replace('74508','102217').replace('124268','102217');t=t.replace("js(D/'COMPLETE.json')","js(D.parent/'COMPLETE.json')").replace("js(D/'REMOTE_CPU_ACCEPTANCE.json')","js(D.parent/'REMOTE_CPU_ACCEPTANCE.json')").replace("['target_loss','source_KL','weight_used','weighted_KL','total_loss']","['target_loss','Omega','scaled_Omega','L2_distance_before_update','total_loss']");t=comparisons(t);ast.parse(t);(T/f'analyze_phase227_{domain}.py').write_text(t,encoding='utf-8')
 t=names((T/f'validate_phase{prior}.py').read_text()).replace(f'phase{prior}/phase{prior}_'+('20261025' if prior==226 else '20261024'),'phase227/phase227_20261025');a=t.index(' clock=time.monotonic();');b=t.index(" strata=js(D/",a) if domain=='l8' else t.index(" base={p:js(D/",a);t=t[:a]+' clock=time.monotonic()\n'+t[b:];t=comparisons(t);t=t.replace("'raw_files_SHA_checked':len(m)","'raw_manifest_checked_in_parent':True").replace("'all16000_weighted_steps_checked':True,'frozen_CSV_and_original_functions':True,",'');ast.parse(t);(T/f'validate_phase227_{domain}.py').write_text(t,encoding='utf-8')
print('two-domain tables and independent math validators generated')
