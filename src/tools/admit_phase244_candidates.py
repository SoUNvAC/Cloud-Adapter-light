"""Bounded metadata acquisition for PixBox L8 / WHU Cloud; no image payloads."""
from pathlib import Path
import urllib.request, json, hashlib, io, zipfile

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/phase244/phase244_20261031'
URLS={
 'pixbox_record.json':'https://zenodo.org/api/records/5040271',
 'PixBox-L8-CMIX.zip':'https://zenodo.org/records/5040271/files/PixBox-L8-CMIX.zip?download=1',
 'whu_page.html':'https://gpcv.whu.edu.cn/data/WHU_Cloud_Dataset.html',
}

def main():
 OUT.mkdir(parents=True,exist_ok=True);receipts=[]
 for name,url in URLS.items():
  dest=OUT/name
  if dest.exists():
   receipts.append({'name':name,'url':url,'bytes':dest.stat().st_size,'reused':True});continue
  try:
   req=urllib.request.Request(url,headers={'User-Agent':'research-metadata-audit/1.0'})
   with urllib.request.urlopen(req,timeout=40) as r:
    data=r.read(5*1024**2+1)
    if len(data)>5*1024**2:raise ValueError('metadata response exceeds 5MiB cap')
    dest.write_bytes(data);receipts.append({'name':name,'url':url,'final_url':r.url,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
  except Exception as e:receipts.append({'name':name,'url':url,'error':str(e)})
 z=OUT/'PixBox-L8-CMIX.zip'
 if z.exists():
  with zipfile.ZipFile(z) as f:
   index=[{'name':i.filename,'bytes':i.file_size,'CRC':i.CRC} for i in f.infolist()]
   (OUT/'ZIP_INDEX.json').write_text(json.dumps(index,indent=2),encoding='utf-8')
   for i in f.infolist():
    if i.filename.lower().endswith('.txt') and i.file_size<500000:
     # Description only; CSV point attributes/label values remain unopened.
     (OUT/Path(i.filename).name).write_bytes(f.read(i))
 (OUT/'ACQUISITION.json').write_text(json.dumps({'requests':receipts,'image_downloads':0,'label_CSV_parsed':False,'new_model_calls':0},indent=2),encoding='utf-8')
 print(json.dumps(receipts,indent=2))

if __name__=='__main__':main()
