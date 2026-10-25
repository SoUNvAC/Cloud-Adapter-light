"""Independent CPU ledger audit from saved cells; no raw model/reference recomputation."""
from collections import defaultdict
from datetime import datetime,timezone
import json
import math
import resource
import shutil
import sys
import time
from phase314_common import *
C=('Surface','Cloud','Shadow')
def num(v):return None if v=='NA' else float(v)
def close(x,y):assert (x is None and y is None) or (x is not None and y is not None and abs(x-y)<1e-9),(x,y)
def main():
    assert sys.platform=='linux';start=time.monotonic();resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
    state=load(ROOT/'STATUS.json');assert state['status']=='statistics_complete_pending_independent_audit';assert not (ROOT/'INDEPENDENT_ACCEPTANCE.json').exists()
    manifest=read(ROOT/'DUAL_LABEL_MANIFEST.csv');assert len(manifest)==60
    lookup={r['product']:r for r in manifest};assert sum(r['range_B']=='True' for r in manifest)==4 and sum(r['range_C']=='True' for r in manifest)==12
    assert sum(r['dataset']=='CALIBRATION' for r in manifest)==10 and sum(r['dataset']=='VALIDATION' for r in manifest)==50
    ab={p:[[0]*3 for _ in range(3)] for p in lookup};ma={p:[[0]*3 for _ in range(3)] for p in lookup}
    for file,target,names in [('ANNOTATOR_CONFUSIONS.csv',ab,('label_A','label_B')),('MAIN_ALT_CONFUSIONS.csv',ma,('main_label','alt_label'))]:
        for row in read(ROOT/file):target[row['product']][int(row[names[0]])][int(row[names[1]])]=int(row['count'])
    for p,row in lookup.items():
        assert sum(map(sum,ab[p]))==int(row['Jmask']);assert int(row['main_valid'])==int(row['Jmask'])+int(row['main_only']) and int(row['alt_valid'])==int(row['Jmask'])+int(row['alt_only'])
        expected=ma[p] if row['main_annotator']=='A' else list(map(list,zip(*ma[p])));assert ab[p]==expected
        assert int(row['main_valid'])+int(row['main_zero_hot'])+int(row['main_multi_hot'])==1022**2
        assert int(row['alt_valid'])+int(row['alt_zero_hot'])+int(row['alt_multi_hot'])==1022**2
    measures={(r['product'],r['reference'],r['strategy']):r for r in read(ROOT/'DUAL_REFERENCE_METRICS.csv') if r['support_kind']=='J' and r['product']!='POOLED4'}
    tensor=defaultdict(list)
    for r in read(ROOT/'TENSOR81.csv'):tensor[r['product']].append((int(r['label_A']),int(r['label_B']),int(r['Source_pred']),int(r['PL_pred']),int(r['count'])))
    changes={(r['product'],r['region']):r for r in read(ROOT/'CHANGE_DECOMPOSITION.csv')};enrichment={r['product']:r for r in read(ROOT/'CHANGE_ENRICHMENT.csv')}
    checks=0
    for p,rows in tensor.items():
        assert len(rows)==81 and len({r[:4] for r in rows})==81
        assert sum(n for a,b,s,q,n in rows)==int(lookup[p]['J'])
        for ref,axis in [('A',0),('B',1)]:
            for arm,predaxis in [('Source',2),('PL',3)]:
                m=[[0]*3 for _ in range(3)]
                for record in rows:m[record[axis]][record[predaxis]]+=record[4]
                row=measures[p,ref,arm];assert m==json.loads(row['confusion']);union=[sum(m[k])+sum(t[k] for t in m)-m[k][k] for k in range(3)]
                assert sum(u>0 for u in union)==int(row['union_classes'])
                for k in range(3):close(num(row[C[k]+'_IoU_pct']),100*m[k][k]/union[k] if union[k] else None)
                valid=[100*m[k][k]/union[k] for k in range(3) if union[k]];close(num(row['mIoU_pct']),sum(valid)/len(valid) if valid else None)
                total=sum(map(sum,m));close(num(row['accuracy_pct']),100*sum(m[k][k] for k in range(3))/total if total else None);checks+=1
        for region in ['E','D']:
            chosen=[r for r in rows if (r[0]==r[1])==(region=='E')];row=changes[p,region];assert sum(r[4] for r in chosen)==int(row['support']);assert sum(n for a,b,s,q,n in chosen if s!=q)==int(row['prediction_changes'])
            for ref,axis in [('A',0),('B',1)]:
                corrected=sum(r[4] for r in chosen if r[2]!=r[axis] and r[3]==r[axis]);damaged=sum(r[4] for r in chosen if r[2]==r[axis] and r[3]!=r[axis]);assert corrected==int(row[ref+'_corrected']) and damaged==int(row[ref+'_damaged']) and corrected-damaged==int(row[ref+'_net_correct'])
            if region=='E':
                assert all(row['A_'+x]==row['B_'+x] for x in ['corrected','damaged','net_correct'])
                assert sum(n for a,b,s,q,n in chosen if s==a and q==a)==int(row['E_both_correct_unchanged'])
                assert sum(n for a,b,s,q,n in chosen if s!=a and q!=a and s==q)==int(row['E_both_wrong_unchanged'])
                assert sum(n for a,b,s,q,n in chosen if s!=a and q!=a and s!=q)==int(row['E_both_wrong_changed'])
            else:
                filters=dict(A_to_B=lambda a,b,s,q:s==a and q==b,B_to_A=lambda a,b,s,q:s==b and q==a,neither_to_A=lambda a,b,s,q:s not in (a,b) and q==a,neither_to_B=lambda a,b,s,q:s not in (a,b) and q==b,A_to_neither=lambda a,b,s,q:s==a and q not in (a,b),B_to_neither=lambda a,b,s,q:s==b and q not in (a,b))
                for name,f in filters.items():assert sum(n for a,b,s,q,n in chosen if s!=q and f(a,b,s,q))==int(row[name])
            checks+=1
        er=enrichment[p];d=int(changes[p,'D']['support']);ch=int(changes[p,'E']['prediction_changes'])+int(changes[p,'D']['prediction_changes']);assert d==int(er['D']) and ch==int(er['prediction_changes'])
        close(num(er['D_fraction_J']),d/int(er['J']) if int(er['J']) else None);close(num(er['D_fraction_changes']),int(changes[p,'D']['prediction_changes'])/ch if ch else None)
        for arm in ['Source','PL']:
            main=lookup[p]['main_annotator'];other='B' if main=='A' else 'A'
            assert measures[p,'main',arm]['confusion']==measures[p,main,arm]['confusion'] and measures[p,'alt',arm]['confusion']==measures[p,other,arm]['confusion']
    pcells=defaultdict(lambda:[[0]*3 for _ in range(3)])
    for row in read(ROOT/'PSEUDO_CONFUSIONS.csv'):pcells[row['product'],row['reference'],row['zone'],row['region']][int(row['truth'])][int(row['teacher_pred'])]=int(row['count'])
    for key,m in pcells.items():
        p,ref,zone,region=key
        if region=='all':assert m==[[pcells[p,ref,zone,'accepted'][i][j]+pcells[p,ref,zone,'rejected'][i][j] for j in range(3)] for i in range(3)]
        if zone=='all':assert m==[[pcells[p,ref,'E',region][i][j]+pcells[p,ref,'D',region][i][j] for j in range(3)] for i in range(3)]
    for row in read(ROOT/'PSEUDO_LABEL_SENSITIVITY.csv'):
        p,ref,z,r=row['product'],row['reference'],row['zone'],row['region'];m=pcells[p,ref,z,r];k=C.index(row['category']);predicted=sum(t[k] for t in m);truth=sum(m[k]);assert predicted==int(row['predicted']) and m[k][k]==int(row['correct']) and truth==int(row['truth_support'])
        close(num(row['precision']),m[k][k]/predicted if predicted else None);den=sum(pcells[p,ref,z,'all'][k]);close(num(row['truth_fraction_within_zone']),truth/den if den else None);checks+=1
    for p in {k[0] for k in pcells}:
        for ref in ['A','B']:assert sum(map(sum,pcells[p,ref,'all','all']))==int(lookup[p]['J'])
        assert pcells[p,'A','E','all']==pcells[p,'B','E','all']
    for path,h in load(ROOT/'INPUT_SHA256.json').items():assert sha(path)==h,path
    elapsed=time.monotonic()-start;mem={k:int(v.split()[0])*1024 for k,v in (l.split(':',1) for l in Path('/proc/meminfo').read_text().splitlines())};assert mem['MemAvailable']>=4000000000 and shutil.disk_usage(ROOT).free>=4000000000
    write(ROOT/'INDEPENDENT_ACCEPTANCE.json',dict(status='passed',utc=datetime.now(timezone.utc).isoformat(),code_sha256=sha(Path(__file__)),ledger_checks=checks,reference_products=60,model_products=4,teacher_products=12,authors_flipped=int(sum(r['main_annotator']=='B' for r in manifest)),raw_arrays_reread=False,all_original_input_SHA_reverified=True,tensor81_and_confusion_margins=True,E_AB_identical=True,alias_not_independent=True,pseudo_region_partitions=True,NA_denominators_preserved=True,seconds=elapsed))
    state.update(status='audited_pending_report',audit_seconds=elapsed,total_seconds=state['seconds']+elapsed,peak_cpu_bytes=max(state['peak_cpu_bytes'],resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),minimum_available_memory=min(state['minimum_available_memory'],mem['MemAvailable']));assert state['total_seconds']<=3600;write(ROOT/'STATUS.json',state)
    print(json.dumps(load(ROOT/'INDEPENDENT_ACCEPTANCE.json')))
if __name__=='__main__':main()
