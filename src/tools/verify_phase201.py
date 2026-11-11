"""Second scalar implementation: no import of replay arithmetic or NumPy."""
import csv,json,math
from pathlib import Path
from prepare_phase201 import O,write
def mean(v):return math.fsum(v)/len(v) if v else None
def metrics(v):
    rows=v['rows'];out={}
    out['macro_AP']=mean([r['AP'] for r in rows if r['positive']>0 and r['negative']>0])
    out['macro_recall']=mean([r['TP']/r['positive'] for r in rows if r['positive']>0]);out['macro_FPR']=mean([r['FP']/r['negative'] for r in rows if r['negative']>0]);out['max_FPR']=max(r['FP']/r['negative'] for r in rows if r['negative']>0)
    out['pooled_AP']=v['pooled_AP'];out['pooled_miou']=None
    if v['pooled_cm'] is not None:
        cm=[[sum(r['cm'][i][j] for r in rows) for j in range(3)] for i in range(3)];assert cm==v['pooled_cm']
        vals=[]
        for i in range(3):
            u=sum(cm[i])+sum(cm[j][i] for j in range(3))-cm[i][i]
            if u:vals.append(cm[i][i]/u)
        out['pooled_miou']=mean(vals)
    return out
def main():
    a=json.loads((O/'NORMALIZED_INPUT.json').read_text(encoding='utf8'))
    with (O/'selection_replay.csv').open(encoding='utf-8-sig') as f:reported=list(csv.DictReader(f))
    checked=0;group_deltas=[];findings=[]
    for block,b in a['blocks'].items():
        s={k:metrics(v) for k,v in b['selection'].items()};e={k:metrics(v) for k,v in b['evaluation'].items()}
        for row in [x for x in reported if x['block']==block and x['status'] in ['executed','pending']]:
            rule=row['rule'];best=max(s[m][rule] for m in b['order']);chosen=next(m for m in b['order'] if s[m][rule]==best);assert row['selected']==chosen
            assert abs(float(row['selection_score'])-best)<1e-12;checked+=2
            dom=[]
            for other in b['order']:
                if other==chosen:continue
                aa=e[chosen];bb=e[other]
                if bb['macro_recall']>=aa['macro_recall']-1e-10 and bb['macro_FPR']<=aa['macro_FPR']+1e-10 and bb['max_FPR']<=aa['max_FPR']+1e-10 and (bb['macro_recall']>aa['macro_recall']+1e-10 or bb['macro_FPR']<aa['macro_FPR']-1e-10 or bb['max_FPR']<aa['max_FPR']-1e-10):dom.append(other)
            assert row['dominators']==(';'.join(dom) or 'none');checked+=1
            if block!='D':
                for prefix,tail in [('mean',False),('tail',True)]:
                    feasible=[m for m in b['order'] if e[m]['macro_FPR']<=.01 and (not tail or e[m]['max_FPR']<=.1)]
                    if not feasible:state='no_feasible_candidate';oracle=None;loss=None
                    else:
                        oracle=max(feasible,key=lambda m:e[m]['macro_recall']);state='feasible' if chosen in feasible else 'budget_violation';loss=(e[oracle]['macro_recall']-e[chosen]['macro_recall'])*100 if chosen in feasible else None
                    assert row[prefix+'_state']==state and row.get(prefix+'_oracle','')==(oracle or '')
                    assert row.get(prefix+'_loss_pp','')=='' if loss is None else abs(float(row[prefix+'_loss_pp'])-loss)<1e-10
                    checked+=3
        for side in ['selection','evaluation']:
            for model,p in b[side].items():
                for r in p['rows']:
                    if r['positive']:assert abs(r['recall']-r['TP']/r['positive'])<=1e-12
                    else:assert r['recall'] is None and r['AP'] is None
                    if r['negative']:assert abs(r['FPR']-r['FP']/r['negative'])<=1e-12
                    checked+=1
        for first in b['order']:
            for second in b['order']:
                if first==second:continue
                rr={x['group_id']:x for x in b['evaluation'][second]['rows']}
                for x in b['evaluation'][first]['rows']:
                    y=rr[x['group_id']];group_deltas.append(dict(block=block,group_id=x['group_id'],first=first,second=second,positive=x['positive'],negative=x['negative'],delta_AP_pp=(y['AP']-x['AP'])*100 if x['AP'] is not None else None,delta_recall_pp=(y['recall']-x['recall'])*100 if x['recall'] is not None else None,delta_FPR_pp=(y['FPR']-x['FPR'])*100 if x['FPR'] is not None else None,delta_miou_pp=(y['miou']-x['miou'])*100 if x['miou'] is not None else None))
    for r in a['met']:
        t=r['t'];p=r['p'];tp=r['TP'];u=[t[i]+p[i]-tp[i] for i in range(3)]
        old=mean([tp[i]/u[i] for i in range(3) if u[i]>0]);new=mean([tp[i]/u[i] for i in range(3) if t[i]>0]);assert abs(old-r['old_mIoU'])<1e-12 and abs(new-r['reference_classes_mIoU'])<1e-12
        assert sum(t)-sum(tp)==r['errors'] and abs(r['errors']/sum(t)-r['error_rate'])<1e-12
        checked+=4
    for phase,order in [('81',['LL','LM','MM','ML']),('84',['LL','R4','F_real','F_shuffle'])]:
        for rule in ['old_mIoU','reference_classes_mIoU']:
            vals={m:mean([r[rule] for r in a['met'] if r['phase']==phase and r['policy']==m]) for m in order};winner=max(order,key=lambda m:vals[m]);assert winner==a['met_rankings'][phase][rule][0]['policy'];checked+=1
    with (O/'paired_group_differences.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(group_deltas[0]));w.writeheader();w.writerows(group_deltas)
    write(O/'INDEPENDENT_REVIEW.json',dict(status='passed',scalar_checks=checked,all_group_delta_rows=len(group_deltas),implementation='standard-library scalar math, independent ranking/dominance/feasibility/loss, no replay arithmetic import',MET_groups_verified=256,rankings_verified=4,new_model_calls=0))
    print('Independent scalar checks passed:',checked)
if __name__=='__main__':main()
