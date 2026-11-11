"""Bounded official text/metadata only; never download arrays, images or weights."""
import hashlib,json,urllib.request,urllib.error,datetime,re
from pathlib import Path
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase211/phase211_20261020';S=O/'sources';BUDGET=20000000;used=0;records=[]
def fetch(name,url,cap=2000000):
 global used
 p=S/name
 assert not p.exists();cap=min(cap,BUDGET-used);assert cap>0
 try:
  req=urllib.request.Request(url,headers={'User-Agent':'Phase211-metadata-audit','Accept-Encoding':'identity'})
  with urllib.request.urlopen(req,timeout=35) as r:
   length=r.headers.get('Content-Length');assert length is None or int(length)<=cap,'declared text exceeds metadata budget';body=r.read(cap+1);assert len(body)<=cap,'metadata byte cap';used+=len(body);p.write_bytes(body);records.append(dict(name=name,url=url,final_url=r.geturl(),bytes=len(body),sha256=hashlib.sha256(body).hexdigest(),取得日期_北京='2026-10-20',retrieved_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='ok',etag=r.headers.get('ETag')));return body
 except Exception as e:
  records.append(dict(name=name,url=url,status='failed',error=str(e),取得日期_北京='2026-10-20'));return None
 finally:
  (O/'SOURCE_FETCH_MANIFEST.json').write_text(json.dumps(dict(records=records,new_text_metadata_bytes=used,cap=BUDGET,images_weights_pixels_downloaded=0),indent=2,ensure_ascii=False),encoding='utf8')
def main():
 O.mkdir(parents=True,exist_ok=True);S.mkdir(exist_ok=False)
 commit=fetch('omni_main_commit.json','https://api.github.com/repos/DPIRD-DMA/OmniCloudMask/commits/main')
 rev=json.loads(commit)['sha'] if commit else 'main'
 fetch('MODEL_CHANGELOG.md',f'https://raw.githubusercontent.com/DPIRD-DMA/OmniCloudMask/{rev}/MODEL_CHANGELOG.md')
 fetch('how-it-works.html','https://omnicloudmask.readthedocs.io/en/latest/how-it-works.html')
 fetch('v4_training_map.html','https://dpird-dma.github.io/OCM-training-data-map/',cap=6000000)
 tree=fetch('v4_map_tree.json','https://api.github.com/repos/DPIRD-DMA/OCM-training-data-map/git/trees/main?recursive=1')
 fetch('cloudsen_high_README.md','https://huggingface.co/datasets/csaybar/CloudSEN12-high/resolve/d97728dd61858a21c7d6bc61385687d328433cba/README.md')
 fetch('test_metadata.csv','https://huggingface.co/datasets/csaybar/CloudSEN12-high/resolve/d97728dd61858a21c7d6bc61385687d328433cba/test/metadata.csv')
 fetch('cloudsen_test_tree.json','https://huggingface.co/api/datasets/csaybar/CloudSEN12-high/tree/d97728dd61858a21c7d6bc61385687d328433cba/test')
 # Fetch at most one map data file, using only the official map's repository inventory.
 if tree:
  d=json.loads(tree);candidates=[x for x in d.get('tree',[]) if x['type']=='blob' and x['path'].endswith(('.geojson','.json')) and x.get('size',0)<=8000000 and 'package' not in x['path']]
  if len(candidates)==1:
   q=candidates[0];fetch('v4_map_data'+Path(q['path']).suffix,f"https://raw.githubusercontent.com/DPIRD-DMA/OCM-training-data-map/{d['sha']}/{q['path']}",cap=8000000)
 print(json.dumps(dict(bytes=used,records=[dict(name=q['name'],status=q['status'],bytes=q.get('bytes')) for q in records])))
if __name__=='__main__':main()
