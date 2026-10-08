"""ALCD intake only: pinned archive -> CDSE catalogue -> original public GCS L1C.

Never changes a research split or author labels. No account, credential or L2A fallback.
Downloads are generation-pinned and length/MD5/SHA256 checked; a live lock is never bypassed.
"""
import argparse
import base64
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request

ARCHIVE_SHA = '5912fbcfe9edbc1c2cdffbb7ef0119ffdb3099bc9eec37efe301929008bf966d'
BANDS = ('B02', 'B03', 'B04', 'B08', 'B11', 'B12')
BUCKET = 'gcp-public-data-sentinel-2'

def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)

def digest(path):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            md5.update(block); sha.update(block)
    return base64.b64encode(md5.digest()).decode(), sha.hexdigest()

def get_json(url):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as f:
                return json.load(f)
        except (OSError, ValueError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)

def listing(prefix, delimiter=None):
    result = {'items': [], 'prefixes': []}
    query = {'prefix': prefix, 'maxResults': 1000}
    if delimiter:
        query['delimiter'] = delimiter
    while True:
        page = get_json('https://storage.googleapis.com/storage/v1/b/' + BUCKET + '/o?' + urllib.parse.urlencode(query))
        for key in result:
            result[key].extend(page.get(key, []))
        if not page.get('nextPageToken'):
            return result
        query['pageToken'] = page['nextPageToken']

def scenes(archive):
    if digest(archive)[1] != ARCHIVE_SHA:
        raise ValueError('ALCD archive SHA256 mismatch')
    rows = {}
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar:
            if member.isfile() and member.name.endswith('/Classification/used_parameters.json'):
                p = json.load(tar.extractfile(member))
                key = p['cloudy_product_name'] + '__' + p['tile']
                if key in rows and rows[key]['parameters'] != p:
                    # Location spelling can differ; identity must not.
                    for field in ('tile','cloudy_date','cloudy_product_name','clear_date','clear_product_name'):
                        if rows[key]['parameters'][field] != p[field]:
                            raise ValueError('Conflicting duplicate parameters')
                rows.setdefault(key, {'key':key, 'parameters':p, 'label_members':[]})['label_members'].append(member.name.replace('used_parameters.json','classification_map.tif'))
    if len(rows) != 37:
        raise ValueError('Unexpected unique ALCD product/tile count')
    return list(rows.values())

def select_original(old, tile, prefixes):
    if '_OPER_PRD_' not in old:
        return [p for p in prefixes if p.endswith('/' + old + '.SAFE/')]
    generation = old.split('_PDMC_')[1].split('_')[0]
    sensing = old.split('_V')[1].split('_')[0]
    orbit = re.search(r'_R\d{3}_', old).group(0)
    # Legacy SAFE names were normalized by Google. Demand the original validity
    # timestamp AND generation timestamp AND orbit, never merely date/tile.
    return [p for p in prefixes if ('_MSIL1C_' + sensing + '_') in p and
            orbit in p and ('_T' + tile + '_') in p and p.endswith('_' + generation + '.SAFE/')]

def discover(archive, root):
    path = root / 'product_manifest.json'
    manifest = json.loads(path.read_text()) if path.exists() else {'created_utc':utc(), 'archive_sha256':ARCHIVE_SHA, 'purpose':'intake_not_training_authorization', 'scenes':[]}
    done = {r['key'] for r in manifest['scenes']}
    for row in scenes(archive):
        if row['key'] in done:
            continue
        p = row['parameters']; tile = p['tile']; date = p['cloudy_date']
        if not re.fullmatch(r'\d{2}[A-Z]{3}', tile) or not re.fullmatch(r'\d{8}', date):
            raise ValueError('Invalid author tile/date')
        start = dt.datetime.strptime(date,'%Y%m%d'); end = start + dt.timedelta(days=1)
        orbit = re.search(r'_R\d{3}_', p['cloudy_product_name']).group(0)
        filt = "Collection/Name eq 'SENTINEL-2' and contains(Name,'MSIL1C') and contains(Name,'_T%s_') and contains(Name,'%s') and ContentDate/Start ge %sZ and ContentDate/Start lt %sZ" % (tile,orbit,start.isoformat(),end.isoformat())
        url = 'https://catalogue.dataspace.copernicus.eu/odata/v1/Products?' + urllib.parse.urlencode({'$filter':filt,'$top':100})
        catalog = get_json(url)
        if catalog.get('@odata.nextLink'):
            raise ValueError('CDSE pagination requires explicit review')
        row.update(cdse_query=url,cdse_response=catalog)
        prefix = 'tiles/' + tile[:2] + '/' + tile[2] + '/' + tile[3:] + '/' + p['cloudy_product_name'][:3] + '_MSIL1C_' + date
        candidates = listing(prefix, '/')['prefixes']
        row['gcs_candidates'] = candidates
        selected = select_original(p['cloudy_product_name'],tile,candidates)
        row['status'] = 'unresolved_original_product'
        if len(selected) == 1:
            objects = listing(selected[0])['items']; chosen = []
            for band in BANDS:
                matches = [o for o in objects if '/IMG_DATA/' in o['name'] and o['name'].endswith('_'+band+'.jp2')]
                if len(matches) != 1:
                    raise ValueError('Ambiguous/missing band: '+row['key']+' '+band)
                chosen.extend(matches)
            xml = [o for o in objects if o['name'].endswith('.xml') and
                   (o['name'].count('/') == selected[0].count('/') or '/GRANULE/' in o['name'])]
            if not xml:
                raise ValueError('Missing product/granule metadata')
            chosen.extend(xml)
            row.update(gcs_product_prefix=selected[0],objects=chosen,status='original_product_objects_found')
        manifest['scenes'].append(row); manifest['updated_utc']=utc(); write(path,manifest)
        print(row['key'],row['status'],'CDSE',len(catalog.get('value',[])),flush=True)
    manifest['discovery_complete']=True
    manifest['resolved_scenes']=sum(r['status']=='original_product_objects_found' for r in manifest['scenes'])
    manifest['total_band_bytes']=sum(int(o['size']) for r in manifest['scenes'] for o in r.get('objects',[]) if o['name'].endswith('.jp2'))
    write(path,manifest)

def download_object(obj, dest):
    expected = int(obj['size']); dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():
        md5, sha = digest(dest)
        if dest.stat().st_size != expected or md5 != obj['md5Hash']:
            raise ValueError('Existing file verification failure: '+str(dest))
        return sha
    partial = dest.with_suffix(dest.suffix+'.partial')
    url = 'https://storage.googleapis.com/'+BUCKET+'/'+urllib.parse.quote(obj['name'],safe='/')+'?generation='+obj['generation']
    for attempt in range(8):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > expected:
            raise ValueError('Oversized partial')
        if offset == expected:
            break
        req = urllib.request.Request(url,headers={'Range':f'bytes={offset}-','Accept-Encoding':'identity'})
        try:
            with urllib.request.urlopen(req,timeout=60) as response:
                if offset and (response.status != 206 or not response.headers.get('Content-Range','').startswith(f'bytes {offset}-')):
                    raise ValueError('Server did not honor exact resume')
                with partial.open('ab') as f:
                    while True:
                        block = response.read(1024*1024)
                        if not block: break
                        f.write(block)
            break
        except OSError:
            if attempt == 7: raise
            time.sleep(min(2**attempt,30))
    md5, sha = digest(partial)
    if partial.stat().st_size != expected or md5 != obj['md5Hash']:
        raise ValueError('Length/MD5 failure (partial retained): '+str(partial))
    partial.replace(dest)
    return sha

def download(root):
    manifest = json.loads((root/'product_manifest.json').read_text())
    if not manifest.get('discovery_complete') or manifest['archive_sha256'] != ARCHIVE_SHA:
        raise ValueError('Discovery not complete or archive identity changed')
    path = root/'imagery_download_status.json'
    status = json.loads(path.read_text()) if path.exists() else {'started_utc':utc(),'files':{}}
    status['status']='downloading'; write(path,status)
    for row in manifest['scenes']:
        for obj in row.get('objects',[]):
            relative = obj['name'].removeprefix('tiles/')
            if '..' in Path(relative).parts or '\\' in relative:
                raise ValueError('Unsafe object path')
            dest = root/'objects'/relative
            sha = download_object(obj,dest)
            status['files'][obj['name']]={'status':'verified','size':int(obj['size']),'md5_base64':obj['md5Hash'],'sha256':sha,'path':str(dest.relative_to(root)),'generation':obj['generation']}
            status['updated_utc']=utc(); write(path,status)
            print('verified',obj['name'],flush=True)
    status['status']='all_resolved_files_verified' if manifest['resolved_scenes']==37 else 'resolved_subset_verified_unresolved_products_remain'
    status['completed_utc']=utc(); write(path,status)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('action',choices=['discover','download']); ap.add_argument('--archive',type=Path); ap.add_argument('--root',type=Path,required=True); a=ap.parse_args()
    root=a.root.resolve(); allowed=Path('D:/Cloud-Adapter-light').resolve() if os.name=='nt' else Path('/home/scv/shared').resolve()
    if not root.is_relative_to(allowed): raise ValueError('Root outside authorized workspace')
    if a.archive and not a.archive.resolve().is_relative_to(allowed): raise ValueError('Archive outside authorized workspace')
    root.mkdir(parents=True,exist_ok=True); lock=root/'IMAGERY_LOCK'; lock.mkdir()
    (lock/'PID').write_text(str(os.getpid()))
    try:
        if a.action=='discover': discover(a.archive,root)
        else: download(root)
    finally:
        (lock/'PID').unlink(); lock.rmdir()

if __name__=='__main__': main()
