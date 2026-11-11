"""Read extensionless original MTL and native mask tags; no remapping."""
from pathlib import Path
import json,tarfile,hashlib,re
import rasterio

ROOT=Path('/home/scv/shared/phase246_20261031')
results=[]
for p in sorted(ROOT.glob('*_ARCHIVE.json')):
 a=json.loads(p.read_text());matches=[m['name'] for m in a['members'] if m['regular_file'] and m['name'].upper().endswith('_MTL')]
 r={'scene':a['scene'],'MTL_members':matches}
 if len(matches)==1:
  with tarfile.open(a['archive'],'r:gz') as t:data=t.extractfile(matches[0]).read()
  text=data.decode('ascii',errors='replace').rstrip('\x00');dest=ROOT/(a['scene']+'_metadata')/Path(matches[0]).name;dest.write_bytes(data)
  r['MTL_sha256']=hashlib.sha256(data).hexdigest();r['MTL_text']=text
  r['fields']={m.group(1):m.group(2).strip().strip('"') for m in re.finditer(r'^\s*([A-Za-z0-9_]+)\s*=\s*(.*?)\s*$',text,re.M)}
 selected=ROOT/(a['scene']+'_native_selected');mask=list(selected.glob('*mask*.TIF'))
 if len(mask)==1:
  with rasterio.open(mask[0]) as ds:
   r['mask_tags']=ds.tags();r['band_tags']=ds.tags(1);r['description']=ds.descriptions;r['colorinterp']=[str(x) for x in ds.colorinterp]
   try:r['colormap_selected']={str(k):v for k,v in ds.colormap(1).items() if k in [0,64,128,192,255]}
   except Exception as e:r['colormap_error']=str(e)
 results.append(r)
(ROOT/'METADATA_RECOVERY.json').write_text(json.dumps({'results':results,'new_model_calls':0,'label_mapping_changed':False},indent=2),encoding='utf-8')
print(json.dumps([{'scene':r['scene'],'fields':r.get('fields'),'mask_tags':r.get('mask_tags'),'band_tags':r.get('band_tags'),'colormap':r.get('colormap_selected')} for r in results]))
