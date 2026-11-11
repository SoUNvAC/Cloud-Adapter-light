"""Retrospective OLD-support selection replay; no new model or test access."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/research_audits/l8_val_test_identity_20261027'


def save(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    metric_file=ROOT/'outputs/phase227/phase227_20261025/TABLE_B_ALL_CANDIDATES.csv'
    with metric_file.open(encoding='utf-8-sig',newline='') as f: allrows=list(csv.DictReader(f))
    rows=[r for r in allrows if r['domain']=='L8' and r['arm']=='C']
    records=[];replay=[]
    for seed in ['21701','21702']:
        candidates=[r for r in rows if r['seed']==seed]
        source=next(r for r in candidates if float(r['alpha'])==0)
        endpoint=next(r for r in candidates if float(r['alpha'])==1)
        # Explicitly retrospective: test was already viewed before this rule replay.
        chosen=sorted(candidates,key=lambda r:(-float(r['val_mIoU']),float(r['alpha'])))[0]
        for r in candidates:
            for key,name in [('val_mIoU','val_yes8'),('test_mIoU','test_all16'),
                             ('test_yes_mIoU','test_yes6'),('test_no_mIoU','test_no10')]:
                records.append(dict(seed=seed,alpha=float(r['alpha']),pool=name,mIoU=float(r[key]),
                                    delta_vs_Source_pp=float(r[key])-float(source[key]),
                                    delta_vs_endpoint_pp=float(r[key])-float(endpoint[key]),
                                    support_status='OLD_FILL_INCLUDED',interpretation='Historical conditional diagnostic'))
        replay.append(dict(seed=seed,rule='RETROSPECTIVE max old val mIoU; tie smaller alpha; five candidates',
                           chosen_alpha=float(chosen['alpha']),chosen_val_mIoU=float(chosen['val_mIoU']),
                           chosen_test_mIoU=float(chosen['test_mIoU']),
                           selected_test_minus_Source_pp=float(chosen['test_mIoU'])-float(source['test_mIoU']),
                           selected_test_minus_endpoint_pp=float(chosen['test_mIoU'])-float(endpoint['test_mIoU']),
                           support_status='OLD_FILL_INCLUDED; NOT CLEAN VALIDATION SELECTION',
                           independent_confirmation=False))
    means=[]
    for alpha in [0,.25,.5,.75,1]:
        for pool in ['val_yes8','test_all16','test_yes6','test_no10']:
            rr=[r for r in records if r['alpha']==alpha and r['pool']==pool]
            assert len(rr)==2
            means.append(dict(alpha=alpha,pool=pool,mIoU=mean(r['mIoU'] for r in rr),
                              delta_vs_Source_pp=mean(r['delta_vs_Source_pp'] for r in rr),
                              delta_vs_endpoint_pp=mean(r['delta_vs_endpoint_pp'] for r in rr),
                              support_status='OLD_FILL_INCLUDED'))
    save('PER_SEED_STRATIFIED_C_GRID.csv',records)
    save('MEAN_STRATIFIED_C_GRID.csv',means)
    save('RETROSPECTIVE_VAL_SELECTED_ALPHA.csv',replay)
    inputs=[metric_file,ROOT/'src/configs/protocol/phase64_target_parent_rgb.py',
            ROOT/'src/tools/summarize_phase64_results.py',ROOT/'src/tools/execute_phase216.py',
            ROOT/'src/research_plans/PHASE64_FINAL_RESULT_AUDIT.md',
            ROOT/'outputs/phase222/phase222_20261024/RESULT.md',
            ROOT/'outputs/phase214/phase214_20261023/RESULT.md',
            ROOT/'outputs/phase222/phase222_20261024/delivery/seed21701/C_TRAIN_COMPLETE.json',
            ROOT/'outputs/phase222/phase222_20261024/delivery/seed21702/C_TRAIN_COMPLETE.json']
    (OUT/'AUDIT.json').write_text(json.dumps({'inputs':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                                            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                            'new_training':0,'new_forwards':0,'new_CI':0,'new_test_pixels_read':0,
                                            'warning':'Retrospective replay on previously viewed OLD-support results. No prospective selection or deployment update.'},indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'endpoint_means':[x for x in means if x['alpha']==1], 'retrospective_replay':replay},indent=2))


if __name__=='__main__':main()
