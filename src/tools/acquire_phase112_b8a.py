"""Acquire only fifteen original generation-bound B8A objects; no labels/models."""
import json,hashlib,base64,os,urllib.request,datetime
from pathlib import Path
from install_phase111_runtime import host_free
R=Path(__file__).resolve().parents[2];B=Path('/home/scv/shared/phase112_20261014');P=Path('/home/scv/shared/phase111_ocm_runtime')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):(B/n).write_text(json.dumps(x,indent=2)+'\n')
def main():
 lock=json.loads((R/'src/research_results/phase112_20261014/PRE_EXECUTION_LOCK.json').read_text())
 for n,h in lock['source_sha256'].items():assert sha(R/n)==h,n
 B.mkdir(exist_ok=False);(B/'DOWNLOAD_LOCK').write_text(str(os.getpid()))
 prior=json.loads((P/'ASSET_BUDGET.json').read_text());budget={'prior_received':prior['actual_received_bytes'],'prior_reserved':prior['reserved_bytes'],'received_bytes':0,'reserved_bytes':0,'cap':4000000000};records={}
 save('DOWNLOAD_BUDGET.json',budget)
 for row in lock['products']:
  if row['product_id']==lock['reuse_product']:continue
  size=row['bytes'];assert budget['prior_reserved']+budget['reserved_bytes']+size<=budget['cap'];assert host_free()-size>=4256000000
  budget['reserved_bytes']+=size;save('DOWNLOAD_BUDGET.json',budget)
  dest=B/'B8A'/row['product_id']/'B8A.jp2';dest.parent.mkdir(parents=True);temp=dest.with_suffix('.partial');h=hashlib.sha256();md=hashlib.md5();n=0
  with urllib.request.urlopen(row['future_media_url'],timeout=90) as response,temp.open('xb') as f:
   assert int(response.headers['Content-Length'])==size
   generation=response.headers.get('x-goog-generation');assert generation==row['generation'],(generation,row['generation'])
   while True:
    chunk=response.read(1024*1024)
    if not chunk:break
    n+=len(chunk);budget['received_bytes']+=len(chunk);assert n<=size and budget['prior_received']+budget['received_bytes']<=budget['cap'];f.write(chunk);h.update(chunk);md.update(chunk)
    save('DOWNLOAD_BUDGET.json',budget)
  assert n==size and base64.b64encode(md.digest()).decode()==row['md5_base64']
  temp.rename(dest);records[row['product_id']]={'path':str(dest),'bytes':n,'generation':row['generation'],'md5_pass':True,'length_pass':True,'sha256':h.hexdigest()}
  save('DOWNLOAD_STATUS.json',{'status':'in_progress','files':records});print('verified',row['product_id'],flush=True)
 save('DOWNLOAD_STATUS.json',{'status':'all_files_verified','files':records});(B/'DOWNLOAD_LOCK').unlink();print('15 B8A verified',flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  import traceback
  if B.exists():(B/'DOWNLOAD_FAILURE.txt').write_text(traceback.format_exc())
  raise
