"""Small public metadata only; no rasters, labels or model requests."""
from pathlib import Path
import urllib.request, json, hashlib
from html.parser import HTMLParser

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/phase245/phase245_20261031'

class Tables(HTMLParser):
 def __init__(self):super().__init__();self.rows=[];self.row=None;self.cell=None;self.link=''
 def handle_starttag(self,tag,attrs):
  a=dict(attrs)
  if tag=='tr':self.row=[]
  elif tag in ('td','th') and self.row is not None:self.cell=[];self.link=''
  elif tag=='a' and self.cell is not None and a.get('href'):self.link=a['href']
 def handle_data(self,s):
  if self.cell is not None:self.cell.append(s)
 def handle_endtag(self,tag):
  if tag in ('td','th') and self.cell is not None:
   self.row.append({'text':' '.join(''.join(self.cell).split()),'href':self.link});self.cell=None
  elif tag=='tr' and self.row is not None:self.rows.append(self.row);self.row=None

def main():
 OUT.mkdir(parents=True,exist_ok=True);receipts=[]
 for name,urls in [('irish.html',['https://landsat.usgs.gov/landsat-7-cloud-cover-assessment-validation-data','https://www.usgs.gov/landsat-missions/landsat-7-cloud-cover-assessment-validation-data']),('whu.html',['https://gpcv.whu.edu.cn/data/WHU_Cloud_Dataset.html'])]:
  p=OUT/name
  for url in urls:
   try:
    req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 research-metadata-audit'})
    with urllib.request.urlopen(req,timeout=30) as f:
     b=f.read(2*1024**2+1)
     if len(b)>2*1024**2:raise ValueError('metadata exceeds2MiB')
     if 'text/html' not in f.headers.get('Content-Type',''):raise ValueError('unexpected content type')
    p.write_bytes(b);receipts.append({'url':url,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()});break
   except Exception as e:receipts.append({'url':url,'error':str(e)})
  if p.exists() and name=='irish.html':
   parser=Tables();parser.feed(p.read_text(encoding='utf-8',errors='replace'))
   (OUT/'IRISH_HTML_TABLES.json').write_text(json.dumps(parser.rows,indent=2),encoding='utf-8')
 (OUT/'ACQUISITION.json').write_text(json.dumps(receipts,indent=2),encoding='utf-8');print(json.dumps(receipts,indent=2))

if __name__=='__main__':main()
