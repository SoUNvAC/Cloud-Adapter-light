"""Local official-row identity lock; no image or label decoding."""
from pathlib import Path
import csv, hashlib, json
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/phase216/phase216_20261023'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p): return list(csv.DictReader(p.open(encoding='utf-8-sig',newline='')))
def main():
    OUT.mkdir(parents=True,exist_ok=False)
    meta=ROOT/'outputs/phase65/source_metadata/acquired'
    paths={'train':meta/'train_metadata.csv','val':meta/'val_metadata.csv','test':ROOT/'outputs/phase211/phase211_20261020/sources/test_metadata.csv'}
    manifest=json.loads((meta/'source_metadata_manifest.json').read_text(encoding='utf-8-sig'))
    auditpath=ROOT/'outputs/phase65/work_dirs/phase65a/source_mapping/mapping_audit.json'
    audit=json.loads(auditpath.read_text(encoding='utf-8-sig'))
    assert audit['status']=='verified_rgb_ordinal_mapping'
    print('mapping keys',list(audit));print('metadata manifest keys',list(manifest))
    tables={s:rows(p) for s,p in paths.items()}
    assert [len(tables[s]) for s in ['train','val','test']]==[8490,535,975]
    assert sha(paths['train'])=='d4a4bb41fba0e39d49328db29868273d093f982e84716d7f2b836e852f5349f4'
    assert sha(paths['val'])=='35e7afd8f5feafa6ce6b04415a3e4426ad147b5beed3f84db68839ac12d29097'
    fetch=json.loads((ROOT/'outputs/phase211/phase211_20261020/SOURCE_FETCH_MANIFEST.json').read_text(encoding='utf-8-sig'))
    records=fetch['records']; rec=next(r for r in records if r['name']=='test_metadata.csv');assert rec['sha256']==sha(paths['test'])
    maps=audit.get('mappings',audit.get('mapping'))
    assert maps is not None and len(maps)==9025
    lookup={(r['split'],int(r['png_ordinal'])):r for r in maps}
    for s in ['train','val']:
        for i,r in enumerate(tables[s]):
            m=lookup[s,i];assert str(m['original_index'])==r['index'] and m['product']==r['s2_id'] and m['exact_rgb_match']
    ids={s:{r['index'] for r in rs} for s,rs in tables.items()}
    assert len(ids['train'])==8490 and not ids['train']&ids['val'] and not ids['train']&ids['test']
    ordered=sorted(enumerate(tables['train']),key=lambda ir:(hashlib.sha256(('phase216-source:'+ir[1]['index']).encode()).hexdigest(),ir[1]['index']))[:256]
    chosen=[dict(r,sample_id=r['index'],ordinal=i,png_name=f'{i}.png',rank_sha256=hashlib.sha256(('phase216-source:'+r['index']).encode()).hexdigest(),png_sha256=lookup['train',i]['png_sha256']) for i,r in ordered]
    relation={}
    for s in ['val','test']:
        relation[s]={'same_product_count':sum(r['s2_id'] in {v['s2_id'] for v in tables[s]} for r in chosen),'same_ROI_count':sum(r['roi_id'] in {v['roi_id'] for v in tables[s]} for r in chosen),'spatial_independence':'Not asserted: matching ROI/product counts are metadata, not full geographic exclusion.'}
    out={'sample_id_definition':'Official published CSV index as decimal string; PNG ordinal is published CSV row, not index.', 'official_commit':'d97728dd61858a21c7d6bc61385687d328433cba','metadata_SHA':{s:sha(p) for s,p in paths.items()},'mapping_audit_sha256':sha(auditpath),'test_metadata_provenance':rec,'selected':chosen,'metadata_relation':relation,'ID_disjoint':True,'test_images_permission':'byte SHA only; no pixel or label decoding','train_labels_permission':'never open','source_prior_budget':'Existing supervised Source: official training pool 8490, parent checkpoint 4000 updates; inherited earlier initialization training cost not reconstructed here.'}
    (OUT/'SOURCE_POOL_REQUEST.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'selected':len(chosen),'relation':relation,'request_sha':sha(OUT/'SOURCE_POOL_REQUEST.json')}))
if __name__=='__main__':main()
