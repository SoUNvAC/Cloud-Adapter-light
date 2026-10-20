"""Fixed original-field error strata; no inference, new labels or sample selection."""
import json
from pathlib import Path
import numpy as np


def analyze(root,delivery,reference,sealed,plan,table):
    lock_path = root/'SUBGROUP_MAPPING_LOCK.json'
    lock = reference.load(lock_path)
    refs_path = reference.R/'data/pixbox_s2_cmix_5036991/phase103_20261013/deduplicated_locations.json'
    assert reference.sha(refs_path)=='b1448f3fbd05c9e77af7ed0ea50dbde4be7480a35a4c532a2fec45789af25f96'
    refs = {r['point_ids'][0]:r for r in reference.load(refs_path)}
    maps = {name:{} for name in reference.NAMES}
    for scene in plan:
        if scene['pool']!='PixBox': continue
        pid = scene['product_id']
        for name in reference.NAMES:
            base = reference.OLD if name in ('R4','E4') else delivery
            path = base/'infer'/name/'scores/PixBox'/pid/'UNLABELLED_POINT_SCORES.json'
            assert reference.sha(path)==sealed[name]['PixBox/'+pid]
            maps[name].update({q['point_id']:q for q in reference.load(path)['points']})
    ids = [pid for pid,q in maps['R4'].items() if q['valid'] and refs[pid]['reference']!='ignore']
    assert len(ids)==8569
    truth = np.array([{'negative_surface':0,'negative_cloud':1,'positive_ground_cloud_shadow':2}[refs[p]['reference']] for p in ids],np.uint8)
    predictions = {name:np.array([maps[name][pid]['parent_class'] for pid in ids],np.uint8) for name in reference.NAMES}
    products = np.array([refs[pid]['product_id'] for pid in ids])
    groups = np.array([refs[pid]['group_id'] for pid in ids])
    rows = []
    field_support = []
    display = dict(R4='R4',E4='E4',B='B-PL',M='M-PL')
    for field,mapping in lock['fields'].items():
        values = []
        for pid in ids:
            attributes = refs[pid]['original_attributes']
            raw = {str(a.get(field,'')).strip() for a in attributes}
            value = next(iter(raw)) if len(raw)==1 else None
            try:
                numeric = float(value)
                code = str(int(numeric)) if numeric.is_integer() else None
            except (TypeError,ValueError): code = None
            values.append(code if code in mapping else None)
        values = np.asarray(values,object)
        total = 0
        for code,label in list(mapping.items())+[(None,'missing/conflicting/unmapped metadata')]:
            choose = values==code if code is not None else np.fromiter((v is None for v in values),bool,count=len(values))
            n = int(choose.sum())
            total += n
            yy = truth[choose]
            record = dict(pool='PixBox',field=field,original_code=code,original_definition=label,
                         source_type=lock['source_type'][field],products=len(set(products[choose])),
                         groups=len(set(groups[choose])),reference_points=n,
                         Surface_support=int((yy==0).sum()),Cloud_support=int((yy==1).sum()),Shadow_support=int((yy==2).sum()))
            for name in reference.NAMES:
                pp = predictions[name][choose]
                for source,target,key in ((0,1,'Surface_to_Cloud'),(0,2,'Surface_to_Shadow'),
                                           (1,0,'Cloud_to_Surface'),(2,0,'Shadow_to_Surface')):
                    count = int(((yy==source)&(pp==target)).sum())
                    denominator = int((yy==source).sum())
                    record[display[name]+'_'+key+'_count'] = count
                    record[display[name]+'_'+key+'_pct'] = count/denominator*100 if denominator else None
            rows.append(record)
        assert total==len(ids)
        field_support.append(dict(field=field,common_reference_points=len(ids),missing_conflicting_or_unmapped=int(sum(v is None for v in values)),
                                  subgroup_count=len(mapping)+1,selection='all original codes plus missing; no outcome selection'))
    table('TABLE5_PIXBOX_ORIGINAL_SUBTYPES.csv',rows)
    table('TABLE5_FIELD_SUPPORT.csv',field_support)
    inventory = []
    for scene in plan:
        if scene['pool']=='PixBox':
            for band in scene['input_bindings']:
                inventory.append(dict(pool='PixBox',product=scene['product_id'],sensor='Sentinel-2',band=band['band'],
                          native_m=abs(band['native_grid']['XDIM']),target_m=abs(band['target_grid']['XDIM']),
                          resampling='bilinear uint16' if band['bilinear_uint16'] else 'native target grid',
                          source_sha256=band['sha256'],B08_substitution=False))
        else:
            for band in ('B04','B03','B8A'):
                inventory.append(dict(pool='Catalogue',product=scene['product_id'],sensor='Sentinel-2',band=band,
                          native_m=None,target_m=20,resampling='existing locked Catalogue subscene; no new resampling',
                          source_sha256=scene['group']['image_member_sha256'],B08_substitution=False))
    table('TABLE5_INPUT_DIFFERENCES.csv',inventory)
    scope = [
      dict(candidate='cloud_shadow_surface_confusion',support='Catalogue 32 groups; PixBox 16 products / 10 groups',
           evidence='TABLE5_CONFUSION_ERRORS.csv; all 71 original metadata codes and missing strata retained in TABLE5_PIXBOX_ORIGINAL_SUBTYPES.csv',
           limitation='Snow/salt/desert names are original metadata, not measured brightness; subtype dimensions overlap; no causal mechanism claim'),
      dict(candidate='cross_sensor_band_mismatch',support='Both pools Sentinel-2; B04/B03/B8A at 20m',
           evidence='TABLE5_INPUT_DIFFERENCES.csv; PixBox B04/B03 native10m bilinear to20m; B8A native20m; no B08 substitution',
           limitation='No cross-sensor validation; Catalogue original native grid/resampling detail not reconstructed by this phase'),
      dict(candidate='annotation_convention_shift',support='Common Surface/Cloud/Shadow parents on historical fixed support',
           evidence='Catalogue three-mask truth; model Thick/Thin merged Cloud; PixBox fixed C/S/U mapping, S2 terrain/tree/building shadow remains surface negative, S3 ground-cloud-shadow positive; 931 ignore points',
           limitation='Catalogue no original thin/thick labels; no new thin-cloud/boundary metrics. Terrain-shadow conventions cannot be proven equivalent across sources. Pool differences do not identify label-shift causality')]
    table('TABLE5_SHIFT_SCOPE.csv',scope)
    (root/'SUBGROUP_ACCEPTANCE.json').write_text(json.dumps(dict(status='passed',mapping_sha256=reference.sha(lock_path),
                all_original_codes=71,rows=len(rows),fixed_reference_points=8569,
                no_phenotype_inferred_from_scores=True,overlapping_strata_not_additive=True),indent=2)+'\n')
