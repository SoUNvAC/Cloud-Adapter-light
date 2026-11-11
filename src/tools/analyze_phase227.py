"""CPU only: evaluate both fixed gammas, keeping target domains separate."""
from pathlib import Path
import importlib,csv,json,time
R=next(p for p in Path(__file__).resolve().parents if (p/'src/tools').is_dir());O=R/'outputs/phase227/phase227_20261025'
def rows(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def main():
 assert json.loads((O/'delivery/REMOTE_CPU_ACCEPTANCE.json').read_text())['status']=='passed';start=time.monotonic()
 files=['TABLE_A_FIXED_POINTS.csv','TABLE_B_ALL_CANDIDATES.csv','ALL_POOL_METRICS.csv','PER_GROUP.csv','ERROR_FLOW_NET_CHANGES.csv','SOURCE_TAIL_GROUPS.csv','FOCUS_ROI0124_ALPHA025_AND1.csv','SOURCE_RULE_ALL_ARMS.csv','FIXED_COMPARISONS.csv','PAIRED_CI.csv','FINITE_ALL200_PAIRS.csv','FINITE_DH_COVERAGE.csv','TABLE_C_DIRECT_ALPHA1.csv','TEACHER_CORRECTNESS_ALPHA1.csv','LOSS_SUMMARY.csv']
 for domain in ['SPARCS','L8']:
  m=importlib.import_module('analyze_phase227_'+domain.lower());m.O=O/domain;m.O.mkdir(exist_ok=True);m.D=O/'delivery'/domain;m.E=m.D/'evaluation';m.main()
 for name in files:
  rr=[{'domain':domain,**v} for domain in ['SPARCS','L8'] for v in rows(O/domain/name)]
  with (O/name).open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for v in rr for k in v)));w.writeheader();w.writerows(rr)
 (O/'ANALYSIS_RECEIPT.json').write_text(json.dumps({'status':'complete','seconds':time.monotonic()-start,'new_model_calls':0,'domains_separate':True,'endpoints_primary':'W1-D,W10-D alpha1','no_best_gamma_selection':True},indent=2))
if __name__=='__main__':main()
