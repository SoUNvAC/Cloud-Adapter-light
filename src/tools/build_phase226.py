"""Generate SPARCS-only225 replication from sealed225 code and original217 kernels."""
from pathlib import Path
import ast
R=Path(__file__).resolve().parents[2];T=R/'src/tools'
def cv(t):
 return t.replace('phase225_20261024','phase226_20261025').replace('phase225','phase226').replace('124268','74508').replace('124269','74509').replace('60252','10492').replace('57570','7810').replace('140000','85000').replace('21600','14400').replace('21660','14460')
def put(n,t):ast.parse(t);(T/n).write_text(t,encoding='utf-8')
for n in ['phase225_weights.py','controller_phase225.py','check_phase225.py','evaluate_phase225.py','collect_phase225.py','receive_phase225.py']:
 t=cv((T/n).read_text(encoding='utf-8-sig'))
 t=t.replace('phase222_20261024','phase217_20261023').replace('original222','original217')
 if n=='evaluate_phase225.py':
  t=t.replace('SOURCE_TO_PARENT,L8_TO_PARENT','SOURCE_TO_PARENT,SPARCS_TO_PARENT').replace('L8_TO_PARENT','SPARCS_TO_PARENT').replace('[535,963,2167]','[535,10,10]').replace('same frozen features as C endpoint','same frozen features as R endpoint')
 if n=='collect_phase225.py':t=t.replace('mapping=np.array([0,2,1,1],np.int64)','mapping=np.array([2,2,0,0,0,1,0],np.int64)').replace('valid=v<4','valid=v<7')
 put(n.replace('225','226'),t)
t=cv((T/'prepare_phase225.py').read_text());t=t.replace("OLD=R/'outputs/phase222/phase222_20261024'","OLD=R/'outputs/phase217/phase217_20261023'").replace('range(217,225)','range(217,226)').replace('train_phase222_seed','train_phase217_seed').replace(".replace('phase222_20261024','phase226_20261025')",".replace('phase217_20261023','phase226_20261025')").replace(".replace('<120000','<85000')",".replace('<60000','<85000')").replace('calls_cap\':120000','calls_cap\':60000').replace('original222','original217').replace('original222','original217').replace('original222','original217')
t=t.replace(" for n in ['INPUT_SUPPORT_LOCK.json','L8_TRAIN_INPUT_LOCK.json','INPUT_PREFLIGHT_ACCEPTED.json']:shutil.copy2(OLD/'delivery'/n,O/n)"," shutil.copy2(OLD/'PHASE216_EVAL_INPUT_LOCK.json',O/'INPUT_SUPPORT_LOCK.json')")
t=t.replace('L8_{s}_H','SPARCS_{s}_H').replace('PHASE222_COMPLETE_COST','PHASE217_COMPLETE_COST').replace('phase222_20261024','phase217_20261023').replace('PHASE225_IMAGE_LEVEL_SOURCE_REPLAY_WEIGHTING_20261024.md','PHASE226_SPARCS_IMAGE_REPLAY_WEIGHTING_REPLICATION_20261025.md').replace('Only L8','Only SPARCS').replace('original222','original217').replace('6hGPU','4hGPU').replace('phase226_frozen_20261024','phase226_frozen_20261025')
put('prepare_phase226.py',t)
# SPARCS has ten whole scenes, no L8 yes/no strata. Keep all remaining math identical.
t=cv((T/'analyze_phase225.py').read_text()).replace('Frozen L8','Frozen SPARCS')
t=t.replace("strata=js(D/'INPUT_PREFLIGHT_ACCEPTED.json')['test_strata'];",'')
t=t.replace("(['full','yes','no'] if p=='test' else ['full'])","['full']")
t=t.replace("[('', 'full'),('_yes','yes'),('_no','no')]","[('', 'full')]")
t=t.replace("'FPR_qualification':'nominal reference; Shadows=no not verified true absence'","'FPR_qualification':'provided SPARCS dense reference'")
t=t.replace("v['target']=='L8'","v['target']=='SPARCS'")
t=t.replace(",'yes':strata['yes'],'no':strata['no']",'').replace("for k in ['full','yes','no']:","for k in ['full']:")
put('analyze_phase226.py',t)
t=cv((T/'validate_phase225.py').read_text()).replace('phase222/phase222_20261024','phase217/phase217_20261023').replace('train_phase222_seed','train_phase217_seed')
t=t.replace("strata=js(D/'INPUT_PREFLIGHT_ACCEPTED.json')['test_strata'];",'').replace('[107,8,16]','[107,10,10]').replace(",'yes':strata['yes'],'no':strata['no']",'')
put('validate_phase226.py',t)
print('Generated SPARCS-only preparation, kernels wrapper, evaluator and local tables')
if __name__=='__main__':pass
