"""Metadata-only admission. Never opens imagery, labels, scores or weights."""
from pathlib import Path
import base64, collections, csv, hashlib, json, shlex, subprocess, time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/phase243/phase243_20261031'
EVIDENCE = {}

def read(rel):
    p = ROOT / rel
    b = p.read_bytes()
    EVIDENCE[rel] = {'sha256': hashlib.sha256(b).hexdigest(), 'bytes': len(b)}
    if p.suffix == '.csv':
        return list(csv.DictReader(b.decode('utf-8-sig').splitlines()))
    return json.loads(b.decode('utf-8-sig'))

def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

REMOTE = r'''
from pathlib import Path
import os,json,collections,hashlib,datetime
root=Path('/home/scv/Cloud-Adapter-light/src/data')
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'read_only':True,
     'image_label_score_payloads_opened':0,'network_calls':0,'roots':{},'metadata':{}}
for name in ['l8_biome','l8_biome_raw','sparcs','sentinel2_cloud_mask_catalogue_4172871','sentinel2_alcd_1460961','pixbox_s2_cmix_5036991']:
 p=root/name
 r={'exists':p.exists(),'resolved':str(p.resolve()),'suffix_counts':{},'top_entries':[],'landsat_scene_ids':[],'visited_files':0,'truncated':False}
 counts=collections.Counter(); scenes=set()
 if p.is_dir():
  r['top_entries']=[x.name for x in sorted(p.iterdir())][:100]
  for parent,dirs,files in os.walk(p,followlinks=False):
   depth=len(Path(parent).relative_to(p).parts)
   if depth>=5:dirs[:]=[]
   for v in dirs+files:
    import re
    scenes.update(re.findall(r'LC8\d{13}LGN\d{2}',v))
   for v in files:counts[Path(v).suffix.lower()]+=1;r['visited_files']+=1
   if r['visited_files']>200000:r['truncated']=True;break
 r['suffix_counts']=dict(counts);r['landsat_scene_ids']=sorted(scenes);out['roots'][name]=r
for rel in ['work_dirs/phase65a/split_lock.json']:
 p=Path('/home/scv/Cloud-Adapter-light/src')/rel
 if p.is_file():
  b=p.read_bytes();v=json.loads(b);out['metadata'][rel]={'sha256':hashlib.sha256(b).hexdigest(),'rows':len(v.get('proposed_rows',[]))}
out['kappa_named_top_entries']=[str(p) for base in [root,Path('/home/scv/shared/data')] if base.is_dir() for p in base.iterdir() if 'kappa' in p.name.lower()]
print(json.dumps(out))
'''

def main():
    start = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    support = read('outputs/label_support_audit/20261028/PER_SCENE_CLASS_SUPPORT.csv')
    status = read('src/research_plans/protocol_data/l8_biome_usgs_shadow_status.csv')
    groups = read('outputs/phase65/source_metadata/scene_groups.json')['groups']
    split = read('outputs/phase65/work_dirs/phase65a/split_lock.json')
    alcd = read('outputs/phase86/phase86_20261011/groups.csv')
    locations = read('outputs/phase102/phase102_20261013/product_location_index.csv')
    read('outputs/phase102/phase102_20261013/metadata_audit.json')
    for rel in ['outputs/phase89/phase89_20261011/recovered_results/per_group.csv',
                'outputs/phase105/phase105_20261013/per_group_results.csv']:
        read(rel)
    candidates = []
    used_l8 = {r['scene']:r['split'] for r in support if r['dataset']=='L8_Biome' and r['support_scope']!='official_metadata_only_no_pixels_read'}
    for r in status:
        seen = r['scene'] in used_l8
        candidates.append(dict(pool='L8',id=r['scene'],unit='scene',role=used_l8.get(r['scene'],'not_in_current_42'),
          status='research_exposed' if seen else 'shadow_coverage_not_qualified',
          reason='current training/validation/test inventory' if seen else 'official Shadows=no; not reliable Shadow-negative support',
          annotation=r['usgs_shadows'],ready=False))
    sp = [r for r in support if r['dataset'].lower().startswith('sparcs')]
    for r in sp:
        candidates.append(dict(pool='SPARCS',id=r['scene'],unit='scene',role=r['split'],status='research_exposed',reason='existing split inventory',annotation='existing native support audit',ready=False))
    assigned = {r['group_id']:r for r in split['proposed_rows']}
    for r in groups:
        a = assigned.get(r['group_id'])
        candidates.append(dict(pool='Catalogue',id=r['group_id'],unit='related_group',role=a['split'] if a else '',
          status='previously_allocated_group' if a else 'prior_admission_excluded',
          reason='additional members cannot create independent groups' if a else ';'.join(r['exclusion_reasons']),
          annotation=str(len(r['shadow_valid_products']))+' shadow-valid products',ready=False))
    for r in alcd:
        candidates.append(dict(pool='ALCD',id=r['group_id'],unit='related_group',role='Phase89+ evaluated',status='research_exposed',reason='Phase86 pending status superseded by Phase89 evaluation',annotation='original/revised label review limits remain',ready=False))
    for r in locations:
        candidates.append(dict(pool=r['source'],id=r['record_key'],unit='acquisition_key' if 'Kappa' in r['source'] else 'product',role='Phase102 metadata record',
          status=('metadata_unresolved' if r['status']=='pending_metadata' else r['status']) if 'Kappa' in r['source'] else 'research_exposed_or_spatial_excluded',
          reason=r['exclusion_reasons'] if 'Kappa' in r['source'] else '16 retained products evaluated since Phase105; other13 spatially excluded',annotation=r['footprint_precision'],ready=False))
    command='/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python -B -c '+shlex.quote('import base64;exec(base64.b64decode('+repr(base64.b64encode(REMOTE.encode()).decode())+').decode())')
    attempts=[]; remote=None
    for i in range(4):
        try:
            p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ConnectionAttempts=1','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=2','gzs',command],capture_output=True,text=True,encoding='utf-8',timeout=90)
            attempts.append({'attempt':i+1,'exit':p.returncode,'stderr':p.stderr[-1000:]})
            if p.returncode==0:remote=json.loads(p.stdout);break
        except (subprocess.TimeoutExpired,ValueError) as e:attempts.append({'attempt':i+1,'error':type(e).__name__})
        if i<3:time.sleep(3)
    save('REMOTE_RECEIPT.json',remote or {'status':'unavailable'})
    save('CONNECTION_ATTEMPTS.json',attempts)
    checks={'l8_used_count':len(used_l8),'l8_remaining':len(status)-len(used_l8),
      'l8_remaining_shadow_status':dict(collections.Counter(r['usgs_shadows'] for r in status if r['scene'] not in used_l8)),
      'sparcs_support_rows':len(sp),'sparcs_unique_scenes':len({r['scene'] for r in sp}),
      'catalogue_all_groups':len(groups),'catalogue_eligible_groups':sum(bool(r['eligible']) for r in groups),
      'catalogue_allocated_groups':len(assigned),'catalogue_eligible_unallocated':[r['group_id'] for r in groups if r['eligible'] and r['group_id'] not in assigned],
      'catalogue_split_counts':dict(collections.Counter(r['split'] for r in assigned.values())),
      'alcd_already_evaluated_groups':len(alcd),'indexed_source_status_counts':dict(collections.Counter(r['source']+'|'+r['status'] for r in locations)),
      'ready_groups':0,'new_training':0,'new_forward':0,'new_pixel_reads':0,'elapsed_seconds':time.monotonic()-start}
    if remote:
        observed=set(remote['roots']['sparcs']['landsat_scene_ids']);known={r['scene'] for r in sp}
        checks['remote_sparcs_named_ids']=len(observed)
        checks['remote_sparcs_ids_outside_support']=sorted(observed-known)
        checks['remote_catalogue_split_matches_local']=remote['metadata'].get('work_dirs/phase65a/split_lock.json',{}).get('sha256')==EVIDENCE['outputs/phase65/work_dirs/phase65a/split_lock.json']['sha256']
    save('SUMMARY.json',checks);save('EVIDENCE.json',EVIDENCE)
    with (OUT/'CANDIDATES.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['pool','id','unit','role','status','reason','annotation','ready']);w.writeheader();w.writerows(candidates)
    print(json.dumps(checks,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
