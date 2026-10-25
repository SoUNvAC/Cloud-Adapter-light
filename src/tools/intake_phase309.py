"""Isolated PDF tooling, exact-version evidence extraction, fixed-member metadata only."""
import os
import sys
import io
import struct
import shutil
import subprocess
import zipfile
import importlib.util
import traceback
import numpy as np
from phase309_common import *
INPUTS={}
def pin(p):INPUTS[str(p)]=sha(p);return INPUTS[str(p)]
def dbf(data):
    count=struct.unpack('<I',data[4:8])[0];header,length=struct.unpack('<HH',data[8:12]);fields=[]
    for off in range(32,header-1,32):
        if data[off]==13:break
        b=data[off:off+32];fields.append(dict(name=b[:11].split(b'\0')[0].decode('ascii',errors='replace'),type=chr(b[11]),length=b[16],decimals=b[17]))
    rows=[]
    for n in range(count):
        b=data[header+n*length:header+(n+1)*length];pos=1;row={'deleted':b[:1]==b'*'}
        for f in fields:row[f['name']]=b[pos:pos+f['length']].decode('utf-8',errors='replace').strip();pos+=f['length']
        rows.append(row)
    return dict(fields=fields,rows=rows)
def main():
    evidence=ROOT/'evidence';evidence.mkdir();(evidence/'pages').mkdir()
    prior=R/'src/research_results/phase308_20261025';manifest=load(prior/'SUMMARY_SHA256.json');pin(prior/'SUMMARY_SHA256.json')
    for n,h in manifest['files'].items():assert pin(prior/n)==h,n
    for n in ('RESULT.md','DOCUMENT_AUDIT.json','IMAGE_REFERENCE_AUDIT.json','PREVIEW_COORDINATES.json','INPUT_SHA256.json','EDITORIAL_RECEIPT.json'):shutil.copyfile(prior/n,evidence/('phase308_'+n))
    old=load(prior/'DOCUMENT_AUDIT.json')
    for n in ('README.pdf','record.json','classification_tags.csv'):assert pin(D/n)==old[n]['sha256'];shutil.copyfile(D/n,evidence/n)
    inventory=dict(executable=sys.executable,PDF_tools={n:shutil.which(n) for n in ('pdftotext','mutool','gs')},PDF_modules={n:bool(importlib.util.find_spec(n)) for n in ('pypdf','fitz','pdfminer','pdfplumber','PyPDF2')},isolated_target=str(ROOT/'toolenv'),production_packages_unchanged=True)
    before=subprocess.check_output([sys.executable,'-m','pip','list','--format=json'],text=True);(ROOT/'PRODUCTION_PACKAGES_BEFORE.json').write_text(before)
    wheels=ROOT/'wheels';wheels.mkdir()
    with (ROOT/'tool_download.log').open('xb') as log:
        subprocess.run([sys.executable,'-m','pip','download','--no-deps','--only-binary=:all:','--no-cache-dir','--retries','0','--timeout','20','--dest',str(wheels),'pypdf==5.9.0','pypdfium2==4.30.0'],check=True,stdout=log,stderr=subprocess.STDOUT)
    files=sorted(wheels.glob('*.whl'));assert len(files)==2 and sum(p.stat().st_size for p in files)<20000000
    with (ROOT/'tool_install.log').open('xb') as log:subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps','--no-compile','--no-cache-dir','--target',str(ROOT/'toolenv')]+list(map(str,files)),check=True,stdout=log,stderr=subprocess.STDOUT)
    after=subprocess.check_output([sys.executable,'-m','pip','list','--format=json'],text=True);assert before==after
    (ROOT/'PRODUCTION_PACKAGES_AFTER.json').write_text(after)
    inventory['wheels']={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files};inventory['download_wheel_bytes']=sum(p.stat().st_size for p in files)
    sys.path.insert(0,str(ROOT/'toolenv'))
    from pypdf import PdfReader
    import pypdfium2 as pdfium
    reader=PdfReader(D/'README.pdf');document=pdfium.PdfDocument(str(D/'README.pdf'));pages=[]
    for i,page in enumerate(reader.pages):
        text=page.extract_text(extraction_mode='layout');out=evidence/'pages'/f'README_page{i+1:02d}.txt';out.write_text('\n'.join(f'L{j+1:04d}: {line}' for j,line in enumerate(text.splitlines()))+'\n',encoding='utf-8')
        png=evidence/'pages'/f'README_page{i+1:02d}.png';document[i].render(scale=1.3).to_pil().save(png)
        pages.append(dict(page=i+1,text=out.relative_to(ROOT).as_posix(),text_sha256=sha(out),png=png.relative_to(ROOT).as_posix(),png_sha256=sha(png),characters=len(text)))
    (evidence/'README_fulltext.txt').write_text('\n\n'.join(f'=== PAGE {i+1} ===\n'+(evidence/'pages'/f'README_page{i+1:02d}.txt').read_text() for i in range(len(pages))),encoding='utf-8')
    write(ROOT/'PDF_EXTRACTION.json',dict(tool='pypdf5.9.0 layout extraction + pypdfium2 4.30.0 rendering, no OCR',README_sha256=sha(D/'README.pdf'),pages=pages));write(ROOT/'ENVIRONMENT_LOCK.json',inventory)
    audit=load(prior/'IMAGE_REFERENCE_AUDIT.json');products=[r['product'] for r in audit];assert len(products)==3
    listings={};members={};spatial=[]
    for name in ('subscenes.zip','masks.zip','alt_masks.zip','shapefiles.zip'):
        with zipfile.ZipFile(D/name) as z:
            listings[name]=z.namelist()
            if name=='shapefiles.zip':
                pin(D/name)
                for pid in products:
                    row=dict(product=pid,members=[])
                    for n in z.namelist():
                        if pid not in n or n.endswith('/'):continue
                        b=z.read(n);h=__import__('hashlib').sha256(b).hexdigest();m=dict(name=n,sha256=h,bytes=len(b));out=evidence/'shapefiles'/n;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(b)
                        if n.lower().endswith('.dbf'):m['attributes']=dbf(b)
                        elif n.lower().endswith('.prj'):m['WKT']=b.decode('utf-8',errors='replace')
                        elif n.lower().endswith('.shp'):
                            assert struct.unpack('>i',b[:4])[0]==9994
                            m['shape_type']=struct.unpack('<i',b[32:36])[0];m['bbox_xy_m']=list(struct.unpack('<4d',b[36:68]));geometry=[];pos=100
                            while pos<len(b):
                                number,length=struct.unpack('>ii',b[pos:pos+8]);content=b[pos+8:pos+8+length*2];typ=struct.unpack('<i',content[:4])[0]
                                if typ==5:
                                    nparts,npoints=struct.unpack('<ii',content[36:44]);parts=list(struct.unpack('<'+'i'*nparts,content[44:44+4*nparts]));points=np.frombuffer(content,dtype='<f8',offset=44+4*nparts,count=npoints*2).reshape(-1,2).tolist();geometry.append(dict(record=number,parts=parts,xy_m=points))
                                else:geometry.append(dict(record=number,type=typ,unsupported=True))
                                pos+=8+length*2
                            m['geometry']=geometry
                        row['members'].append(m)
                    spatial.append(row)
            elif name in ('subscenes.zip','masks.zip'):
                for pid in products:
                    n=('subscenes/' if name.startswith('sub') else 'masks/')+pid+'.npy';b=z.read(n);arr=np.load(io.BytesIO(b));h=__import__('hashlib').sha256(b).hexdigest()
                    previous=next(r for r in audit if r['product']==pid);assert h==previous['image_sha256' if name.startswith('sub') else 'mask_sha256']
                    members[n]=dict(archive=str(D/name),sha256=h,shape=list(arr.shape),dtype=str(arr.dtype),tool='numpy1.26.4')
                    if name=='masks.zip':members[n]['unique']=np.unique(arr).tolist();members[n]['one_hot_all']=bool(np.isin(arr,[0,1]).all() and (arr.sum(-1)==1).all())
    write(ROOT/'ARCHIVE_LISTINGS.json',listings);write(ROOT/'FIXED_MEMBERS.json',members);write(ROOT/'SHAPEFILE_METADATA.json',spatial)
    for n in ('src/tools/phase302_images.py','src/tools/prepare_phase65a_explore.py','src/tools/download_phase65_sentinel.py'):
        pin(R/n);out=evidence/'project_readers'/Path(n).name;out.parent.mkdir(exist_ok=True);shutil.copyfile(R/n,out)
    record=load(D/'record.json');write(ROOT/'SOURCE_RECORD.json',dict(id=record['id'],doi=record['doi'],created=record['created'],modified=record['modified'],revision=record['revision'],description=record['metadata']['description'],links=record['links']))
    write(ROOT/'INPUT_SHA256.json',INPUTS);write(ROOT/'FAILURES.json',[])
    print((evidence/'README_fulltext.txt').read_text())
if __name__=='__main__':
    try:main()
    except BaseException:write(ROOT/'FAILURES.json',dict(stage='intake',traceback=traceback.format_exc(),no_blind_retry=True));raise
