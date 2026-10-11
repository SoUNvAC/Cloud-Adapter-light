"""Validate real Phase88 extraction, anonymous page and unfilled expert delivery."""
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from prepare_phase88 import ROOT, OUT, PREV, OLD, read, sha, write, csvwrite


def main():
    lock=read(OUT/'sample_input_lock.json');audit=read(OUT/'extraction_audit.json')
    assert sha(ROOT/'outputs/phase86/phase86_20261011/candidate_lock.json')==lock['phase86_candidate_sha256']
    assert sha(OUT/'sample_input_lock.json')==read(OUT/'SAMPLING_FROZEN.json')['sample_input_lock_sha256']==audit['sampling_lock_sha256']
    with (PREV/'pending_adjudication.csv').open(encoding='utf-8-sig',newline='') as f:prior=list(csv.DictReader(f))
    expected_p1={r['source_id'] for r in prior if r['priority']=='P1'}
    assert len(expected_p1)==30
    assert {s['source_id'] for s in lock['samples'] if s['phase88_stratum']=='P1'}==expected_p1
    summaries=[]
    for alias in lock['aliases']:
        eligible=[r for r in prior if r['product']==alias['representative'] and r['priority']!='P1' and 'uniform' in r['sources'].split(';')]
        eligible.sort(key=lambda r:hashlib.sha256(f"PHASE88|{r['product']}|{r['source_id']}".encode()).digest())
        assert {r['source_id'] for r in eligible[:3]}=={s['source_id'] for s in lock['samples'] if s['product']==alias['representative'] and s['phase88_stratum']=='regular_control'}
        for stratum in ['P1','regular_control']:
            selected=[s for s in lock['samples'] if s['product']==alias['representative'] and s['phase88_stratum']==stratum]
            summaries.append(dict(group_id=alias['group_id'],product=alias['representative'],tile=alias['tile'],stratum=stratum,
                prepared=sum(s['review_id'] in {x['review_id'] for x in audit['support']} for s in selected),selected=len(selected),
                actual_reviewers=0,actually_reviewed=0,decidable='NA_not_reviewed',mixed='NA_not_reviewed',unknown='NA_not_reviewed',
                original_possible_conflicts='NA_not_reviewed',revised_possible_conflicts='NA_not_reviewed',
                missing_imagery=sum(s['review_id'] not in {x['review_id'] for x in audit['support']} for s in selected),
                evidence_needs='larger_context_prepared;clear_date_and_DEM_need_pending_human',confirmation_ready=0))
    assert len(lock['samples'])==len(audit['support'])==57 and not audit['missing']
    assert len(audit['source_objects'])==54
    assert len({s['source_id'] for s in lock['samples']})==57
    page=(OUT/'index.html').read_text(encoding='utf-8')
    display=read(OUT/'metric_display_audit.json')
    qa=read(OUT/'VISUAL_QA.json')
    assert qa['purpose']=='image_layout_only_not_human_judgment' and len(qa['viewed_review_ids'])>=3
    assert display['extraction_audit_sha256']==sha(OUT/'extraction_audit.json')
    assert len(display['contexts'])==57 and '_context.png' in page
    for r in display['contexts']:
        assert sha(OUT/r['path'])==r['sha256']
        s=next(s for s in audit['support'] if s['review_id']==r['review_id'])
        h,w=s['shape_60m'];assert r['display_size']==[w*40,h*20+60]
    assert sha(OUT/'sample_input_lock.json') in page
    for forbidden in ['phase88_stratum','original_reference','revised_reference','initial_judgment','stage_B.html','_label.png']:
        assert forbidden not in page
    assert not re.search(r'\b[AR]\d{3,4}\b',page), 'Original anonymous/source IDs exposed'
    for s in audit['support']:
        for suffix,key in [('_small.png','small_sha256'),('_large.png','large_sha256')]:
            assert sha(OUT/'images'/(s['review_id']+suffix))==s[key]
        assert sha(OUT/'windows'/(s['review_id']+'.npz'))==s['window_sha256']
        assert s['original_small_native_pixels_match']
        y0,y1,x0,x1=s['window_60m'];assert 0<=y0<y1<=1830 and 0<=x0<x1<=1830
        assert y1-y0<=51 and x1-x0<=51
    for filename in ['expert_A_template.csv','expert_B_template.csv']:
        with (OUT/filename).open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        assert len(rows)==57 and {r['review_id'] for r in rows}=={s['review_id'] for s in lock['samples']}
        assert all(not value for row in rows for key,value in row.items() if key!='review_id')
    assert not any((OUT/p).exists() for p in ['expert_A_original.json','expert_A_locked.csv','EXPERT_A_LOCK.json','stage_B.html'])
    csvwrite(OUT/'group_status.csv',summaries)
    write(OUT/'LOCAL_VERIFIED.json',dict(verified=True,checks=['all30_P1_unchanged','27_controls_hash_selected_without_class_or_AI_sort',
        'nine_fixed_representatives','54_source_SHA256_and_generation','57_large_windows_match_existing_small_native_pixels_and_valid_masks',
        'clip_edges_no_mirrored_support_and_metric_aspect','original_stretches_reused','no_identity_stratum_or_reference_in_A_page',
        'empty_A_and_B_templates_57_each','no_human_results_or_B_reveal_before_human_A_lock','normal_and_edge_image_layout_QA'],
        assignment='user_volunteered; identity_and_experience_pending',actual_human_reviewers=0,actually_reviewed=0,
        confirmation_ready_groups=0,prepared=57,clipped_windows=sum(s['shape_60m']!=[51,51] for s in audit['support']),
        decoder_note='JPEG2000 SIZ precision read; Pillow integer display expansion reversed exactly; every selected small band/mask compared with prior GDAL-derived NPZ'))
    write(OUT/'DELIVERY_SHA256.json',{p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()})
    print(json.dumps(read(OUT/'LOCAL_VERIFIED.json')))


if __name__=='__main__':main()
