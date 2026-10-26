"""Preserve existing overlap metadata and confirm old-manifest image bytes match shared assets."""
from phase317_common import *
def main():
 resource_check();assert not (ROOT/'HISTORY_IDENTITY.json').exists();out=[];copy=ROOT/'history';copy.mkdir();pins={}
 for name in ['spatial_overlap.json','historical_overlap_v2.json']:
  p=R/'src/work_dirs/phase65a/source_provenance'/name;assert p.exists();h=sha(p);shutil.copyfile(p,copy/name);pins[str(p)]=h
 same=[];scenes={p:{split:sorted({r.get('scene',r.get('group_id')) for r in inputs(p)[split]}) for split in ['fit','val']} for p in ['SPARCS','L8']}
 for pool in ['SPARCS','L8']:
  for split in ['fit','val']:
   for row in inputs(pool)[split]:
    p=Path('/home/scv/shared/data/sparcs/extracted/sending')/Path(row['image_path']).name if pool=='SPARCS' else Path('/home/scv/shared')/row['manifest_image_path']
    exists=p.exists();h=sha(p) if exists else None;assert exists and h==row['image_sha256'],str(p)+' sharedcopy differs from frozenhistoricalmanifest'
    same.append(dict(pool=pool,split=split,name=row['name'],historical_manifest_path=row['image_path'],shared_replica_path=str(p),sha256=h,bytes_match=True))
 table(ROOT/'SHARED_IMAGE_IDENTITY.csv',same)
 inter=[]
 for a in ['fit','val']:
  for b in ['fit','val']:inter.append(dict(SPARCS_split=a,L8_split=b,shared_parent_scene_IDs=sorted(set(scenes['SPARCS'][a])&set(scenes['L8'][b]))))
 write(ROOT/'HISTORY_IDENTITY.json',dict(status='passed',input_metadata_SHA=pins,existing_overlap_metadata_preserved=True,same_parent_ID_intersections=inter,limits='exactparentID intersection only; absence does not prove geographic/pretrainingindependence; historicalmetadata retained not recomputed; no newpool no targetreferencepixels',historical_exposure='CMC/SPARCS/L8 all developed; Phase64 and216-231 supervised/head diagnostics not used as SFDA endpoints',shared_image_matches=len(same),new_neural_calls=0));print('Phase317 shared replica identities and existing overlap records preserved; no referencepixels')
if __name__=='__main__':main()
