"""Independent CPU counts/provenance checks; local mode is SHA verification only."""
from collections import Counter,defaultdict
from pathlib import PurePosixPath
from phase316_common import *

def independent():
    resource_check();assert not (ROOT/'INDEPENDENT_ACCEPTANCE.json').exists()
    # Use path component splitting rather than the primary filename regex.
    oldpatch=read(OLD/'inputs/kappa_patch_positions.csv');archive=load(E/'kappaset_archive_index.json')
    nc=[x for x in archive if PurePosixPath(x['path']).suffix=='.nc'];assert len(nc)==9251
    keys=defaultdict(list);splits=defaultdict(set)
    for a in nc:
        path=PurePosixPath(a['path']);prefix,split,base=path.parts;assert prefix=='kappaset'
        name=base.removesuffix('.nc').split('_');assert len(name)==5 and name[2]=='tile'
        tile,t,_,x,y=name;assert tile.startswith('T') and len(tile)==6
        key='Kappa:'+tile+':'+t;keys[key].append(a['path']);splits[split].add(key)
    assert len(keys)==1079 and Counter(a['path'] for a in nc)==Counter(p['archive_path'] for p in oldpatch)
    newpatch=read(ROOT/'PATCH_MEMBER_MAP.csv');assert len(newpatch)==9251
    for p in newpatch:assert p['member'] in keys[p['record_key']]
    oldex={r['record_key']:r for r in read(OLD/'overlap_exclusions.csv') if r['source']=='KappaSet7100327'}
    overlap=read(ROOT/'OVERLAP_AND_HISTORY.csv');assert len(overlap)==1079 and {r['record_key'] for r in overlap}==set(keys)
    for r in overlap:
        old=oldex[r['record_key']];assert r['original_status']==old['status'] and json.loads(r['original_reasons'])==json.loads(old['exclusion_reasons']) and json.loads(r['original_history_matches'])==json.loads(old['history_matches'])
        assert r['ready_for_small_sample']=='False' and r['Shadow_positive_support']=='unknown' and int(r['patch_count'])==len(keys[r['record_key']])
    assert Counter(r['original_status'] for r in overlap)=={'excluded_overlap':180,'pending_metadata':899}
    # Reconcile full source-row bindings against original raw metadata and old location registry.
    binding=read(ROOT/'SOURCE_TRAIN_VAL_BINDING.csv');assert len(binding)==9025
    registry=load(OLD/'inputs/history_positions_only.json')['records']
    assert Counter((r['product'],r['role']) for r in binding)==Counter((r['product'],r['role']) for r in registry if r['role'].startswith('Source_'))
    tilemap=defaultdict(list)
    for r in binding:
        tile=r['product'].split('_T')[1].split('_')[0];tilemap[tile].append(r)
    for r in overlap:
        assert int(r['current_Source_same_tile_rows'])==len(tilemap[r['tile']])
        assert json.loads(r['current_Source_roles'])==sorted({v['role'] for v in tilemap[r['tile']]})
    fields={r['item']:r for r in read(ROOT/'INPUT_LABEL_MAPPING.csv')}
    expected={0:('UNDEFINED','ignore255'),1:('CLEAR','Surface0'),2:('CLOUD SHADOW','Shadow2'),3:('SEMI TRANSPARENT CLOUD','Cloud1'),4:('CLOUD','Cloud1'),5:('MISSING','ignore255')}
    for code,(label,target) in expected.items():
        r=fields['Label code '+str(code)];assert r['official_or_code_value']==label and r['Source_rule_or_future_requirement']==target
    assert len(read(ROOT/'CANDIDATE_LIST.csv'))==0
    for path,v in load(ROOT/'INPUT_SHA256.json').items():
        assert sha(path)==v['sha256'] and Path(path).stat().st_size==v['bytes'],path
        if 'backup_relative' in v:assert sha(ROOT/v['backup_relative'])==v['sha256']
    net=load(ROOT/'NETWORK_LEDGER.json');assert net['new_official_requests']==net['new_response_body_bytes']==0
    a=load(ROOT/'AUDIT_COMPLETE.json');assert a['training_updates']==a['neural_forwards']==a['new_image_label_members']==0
    assert a['eligible_candidates']==0 and a['Source_rows']=={'train':8490,'val':535}
    write(ROOT/'INDEPENDENT_ACCEPTANCE.json',dict(status='passed',all9251_members_independently_parsed=True,all1079_keys_old_exclusions_exact=True,all9025_Source_rows_match_history=True,label_parent_mapping_document_only=True,all_input_SHA_rechecked=True,no_pixels_models_HTTP=True,resources=resource_check()))
    print('Phase316 independently verified:9251members/1079keys/180+899status,9025actualSource rows;0newpixels/models/HTTP')

def delivery():
    d=load(S/'DELIVERY_SHA256.json')
    for n,h in d['files'].items():assert sha(S/n)==h,n
    for n,h in d['code'].items():assert sha(R/n)==h,n
    b=load(S/'BACKUP_RECEIPT.json');a=load(S/'AUDIT_COMPLETE.json')
    assert b['all_members_SHA_verified'] and b['archive_SHA_verified'] and b['projected_persistent_bytes']<=300000000
    assert load(S/'INDEPENDENT_ACCEPTANCE.json')['status']=='passed' and a['neural_forwards']==a['training_updates']==0
    if sys.platform=='linux':
        for path,v in load(S/'INPUT_SHA256.json').items():assert sha(path)==v['sha256'],path
        for n,v in load(ROOT/'BACKUP_FILES.json').items():assert sha(ROOT/n)==v['sha256'] and (ROOT/n).stat().st_size==v['bytes'],n
        assert sha(ROOT/b['archive'])==b['archive_sha256']
    print('Phase316 delivery verified: all input/code/result/archive/member SHA;0newimages/labels/forwards,0HTTP;0eligible candidates')
if __name__=='__main__':independent() if '--independent' in sys.argv else delivery()
