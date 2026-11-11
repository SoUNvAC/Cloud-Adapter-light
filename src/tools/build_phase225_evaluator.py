"""Derive L8 R/Q fixed evaluator with same-feature D/R/Q endpoint accounting."""
from pathlib import Path
import ast
R=Path(__file__).resolve().parents[2]
def main():
 t=(R/'src/tools/evaluate_phase222.py').read_text().replace('phase222','phase225').replace('<120000','<140000').replace("prior['units']==40010","prior['units']==64016").replace("['C','D']","['R','Q']").replace("seed,'D',1","seed,'Q',1").replace("seed,'C',1","seed,'R',1").replace("arm=='C'","arm=='R'").replace("a=='D'","a=='Q'").replace("a=='C'","a=='R'").replace('candidate_shared_D_head','candidate_shared_Q_head')
 t=t.replace("caches['source']=js(ROOT/'EXPECTED_Source__source.json')","caches['source']=js(ROOT/'historical/evaluation/Source__source.json')").replace("path=ROOT/f'EXPECTED_Source__{pool}.json'","path=ROOT/f'historical/evaluation/Source__{pool}.json'")
 t=t.replace("teacher=copy.deepcopy(model.decode_head).eval();partner=copy.deepcopy(model.decode_head).eval()","teacher=copy.deepcopy(model.decode_head).eval();partner=copy.deepcopy(model.decode_head).eval();oldpartner=copy.deepcopy(model.decode_head).eval();oldmeta=js(ROOT/'EXPERIMENT_LOCK.json')['historical_D'];oldheads={};old_expected={};oldhash={};oldcm={}\n  for s in SEEDS:\n   assert sha(oldmeta[str(s)]['path'])==oldmeta[str(s)]['SHA'];oldheads[s]=torch.load(oldmeta[str(s)]['path'],map_location='cpu')['state_dict'];old_expected[s]=js(ROOT/f'historical/evaluation/{s}_D_a1__source.json');oldhash[s]=hashlib.sha256();oldcm[s]=collections.defaultdict(lambda:np.zeros((3,3),np.int64))")
 t=t.replace('for h in [teacher,partner]:','for h in [teacher,partner,oldpartner]:')
 t=t.replace("partner.load_state_dict(heads[seed,'Q',1]);partner.eval();dsem=semantic(partner,features,'candidate_shared_Q_head');dp=predict(dsem)","partner.load_state_dict(heads[seed,'Q',1]);partner.eval();dsem=semantic(partner,features,'candidate_shared_Q_head');dp=predict(dsem);oldpartner.load_state_dict(oldheads[seed]);oldpartner.eval();oldsem=semantic(oldpartner,features,'historical_D_head');oldp=predict(oldsem)")
 gate="sethead(heads[seed,'R',1]);shared_gates.append({'seed':seed,'name':r['name'],'semantic_bitwise_equal':True})"
 repl="sethead(oldheads[seed]);count('historical_D_gate_full');oldfull=wrapper(x);captured.clear();assert torch.equal(oldfull,oldsem);sethead(heads[seed,'R',1]);shared_gates.append({'seed':seed,'name':r['name'],'Q_and_D_semantic_bitwise_equal':True})"
 assert gate in t;t=t.replace(gate,repl)
 start=t.index('    J=np.bincount(');end=t.index('    pair_data[',start)
 t=t[:start]+'''    oldc=np.bincount(3*y[valid].astype(np.int64)+oldp[valid],minlength=9).reshape(3,3);expected=next(v for v in old_expected[seed]['per_input'] if v['name']==r['name']);assert oldc.tolist()==expected['confusion']
    if 'prediction_SHA' in expected:assert rawsha(oldp)==expected['prediction_SHA']
    oldhash[seed].update(oldp.tobytes());oldcm[seed][r['scene']]+=oldc
    J=np.bincount(81*y[valid].astype(np.int64)+27*tp[valid].astype(np.int64)+9*oldp[valid].astype(np.int64)+3*pred[valid].astype(np.int64)+dp[valid],minlength=243).reshape(3,3,3,3,3)
    joint_stream.write(json.dumps({'seed':seed,'name':r['name'],'ROI':r['scene'],'valid_pixels':int(valid.sum()),'J':J.tolist(),'teacher_prediction_SHA':rawsha(tp),'D_prediction_SHA':rawsha(oldp),'R_prediction_SHA':rawsha(pred),'Q_prediction_SHA':rawsha(dp)})+'\\n');joint_rows+=1
''' +t[end:]
 t=t.replace("save(f'{s}_D_a1__source.json',{'seed':s,'arm':'D'","save(f'{s}_Q_a1__source.json',{'seed':s,'arm':'Q'")
 t=t.replace("assert joint_rows==1070 and not pair_data;save('SHARED_PATH_GATES.json',shared_gates)","assert joint_rows==1070 and not pair_data;save('SHARED_PATH_GATES.json',shared_gates)\n  reproduction={}\n  for s in SEEDS:\n   assert {g:c.tolist() for g,c in oldcm[s].items()}==old_expected[s]['per_group'] and oldhash[s].hexdigest()==old_expected[s]['prediction_SHA'];reproduction[str(s)]={'all535_input_and107_ROI_confusions_exact':True,'prediction_SHA':oldhash[s].hexdigest(),'expected_SHA':old_expected[s]['prediction_SHA']}\n  save('HISTORICAL_D_REPRODUCTION.json',reproduction)")
 t=t.replace("'shared_D_gate_full':2}","'shared_D_gate_full':2,'historical_D_gate_full':2,'historical_D_head':1070}")
 # Q full wrapper check retains the historical counter label, exposed explicitly in the cost receipt.
 ast.parse(t);(R/'src/tools/evaluate_phase225.py').write_text(t,encoding='utf-8')
if __name__=='__main__':main()
