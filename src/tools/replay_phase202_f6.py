"""F6: reuse only accepted per-seed group deltas; no new seeds or intervals."""
import csv,hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase202/phase202_20261017'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert not (O/'F6_ALL_GROUPS.csv').exists()
    paths=[R/'src/research_results/phase124_20261017'/(pool+'_group_evidence.csv') for pool in ['Catalogue','PixBox']]
    lock=dict(files={str(p.relative_to(R)).replace('\\','/'):sha(p) for p in paths},code_sha256=sha(Path(__file__)),no_new_bootstrap=True)
    (O/'F6_INPUT_LOCK.json').write_text(json.dumps(lock,indent=2)+'\n',encoding='utf8')
    out=[]
    for pool,p in zip(['Catalogue','PixBox'],paths):
        with p.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        groups={}
        for v in rows:groups.setdefault(v['group_id'],{})[v['arm']]=v
        for gid,arms in groups.items():
            a,b=arms['C122'],arms['C123'];assert a['positive']==b['positive'] and a['negative']==b['negative']
            yes=int(a['positive'])>0;d1=float(a['AP_delta_vs_R_pp']) if yes else None;d2=float(b['AP_delta_vs_R_pp']) if yes else None
            repeat='NA_no_positive_support' if not yes else 'both_damage' if d1< -1 and d2< -1 else 'both_improve' if d1>1 and d2>1 else 'both_stable' if abs(d1)<=1 and abs(d2)<=1 else 'mixed_direction'
            out.append(dict(pool=pool,group_id=gid,tiles=a['tiles'],products=a['products'],positive=int(a['positive']),negative=int(a['negative']),C122_minus_R_AP_pp=d1,C123_minus_R_AP_pp=d2,two_seed_AP_class=repeat,
                C122_minus_R_recall_pp=a['cal_recall_delta_vs_R_pp'],C123_minus_R_recall_pp=b['cal_recall_delta_vs_R_pp'],C122_minus_R_FPR_pp=a['cal_FPR_delta_vs_R_pp'],C123_minus_R_FPR_pp=b['cal_FPR_delta_vs_R_pp'],
                calibration_C122=a['calibration_threshold'],calibration_C123=b['calibration_threshold'],calibration_R=a['R_calibration_threshold'],sparse_note='7 positives; retained, not sole mechanism evidence' if pool=='PixBox' and '54TWN' in a['tiles'] else ''))
    assert len(out)==42
    with (O/'F6_ALL_GROUPS.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    targets=[x for x in out if x['two_seed_AP_class']=='both_damage'];assert len(targets)==4 and {x['tiles'] for x in targets}=={'32TML','40XDR','19FDV','54TWN'}
    text='| 池／组 | 正参考数 | C122−R AP差pp | C123−R AP差pp | 两seed重复损伤？ | 两seed召回差pp | 两seed误报差pp |\n|---|---|---|---|---|---|---|\n'
    for x in out:
        num=lambda v:'NA' if v is None or v=='' else f'{float(v):.6f}'
        text+='| '+' | '.join([x['pool']+'/'+x['tiles'],str(x['positive']),num(x['C122_minus_R_AP_pp']),num(x['C123_minus_R_AP_pp']),x['two_seed_AP_class'],num(x['C122_minus_R_recall_pp'])+' / '+num(x['C123_minus_R_recall_pp']),num(x['C122_minus_R_FPR_pp'])+' / '+num(x['C123_minus_R_FPR_pp'])])+' |\n'
    text+='\n仅复用R成员C100适配两seed；不是冻结E训练seed损伤，也不代表E适配或V4。AP无正参考填NA；54TWN仅7正参考。区间仅引用Phase124原区间，未重抽。\n'
    (O/'TABLE3_F6.md').write_text(text,encoding='utf8')
    (O/'F6_RECEIPT.json').write_text(json.dumps(dict(status='completed_reused_historical_results',all_groups=42,repeat_damage_groups=targets,model_calls=0,fits=0,new_bootstrap=0),indent=2)+'\n',encoding='utf8')
    print('F6 completed:42 groups retained,4 repeated AP damage groups; no neural calls, fits or new intervals.')
if __name__=='__main__':main()
