"""Offline independent 126-wheel directory pricing and dependency-closure audit."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import struct
import zipfile
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gate-directory',type=Path,required=True)
    base=ap.parse_args().gate_directory
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    resolution=read(base/'PIP_RESOLUTION.json');price=read(base/'complete_wheel_pricing/COMPLETE_ENVIRONMENT_PRICE.json')
    request=read(base/'RESOLUTION_REQUEST.json')
    items={canonicalize_name(x['metadata']['name']):x for x in resolution['install']}
    old=base.parent/'resource_gate_20261014_probe2'
    totals=[]
    for row in price['packages']:
        candidates=list((base/'complete_wheel_pricing').glob(row['name']+'.central-directory.bin'))+list(old.glob('wheel_pricing*/'+row['name']+'.central-directory.bin'))
        matching=[p for p in candidates if hashlib.sha256(p.read_bytes()).hexdigest()==row['central_directory_sha256']]
        assert matching,row['name']
        cd=matching[0].read_bytes();n=row['wheel_entries']
        end=struct.pack('<4s4H2LH',b'PK\x05\x06',0,0,n,n,len(cd),0,0)
        with zipfile.ZipFile(io.BytesIO(cd+end)) as z:total=sum(x.file_size for x in z.infolist())
        assert total==row['uncompressed_all_entries_bytes']
        item=items[canonicalize_name(row['name'])]
        assert item['download_info']['url']==row['url']
        totals.append(total)
    assert sum(totals)==price['all_wheel_uncompressed_declared_bytes'] and len(totals)==len(items)==126
    active={};pending=[Requirement(x) for x in request['requirements']];checked=[]
    while pending:
        req=pending.pop();name=canonicalize_name(req.name)
        assert name in items,name
        assert Version(items[name]['metadata']['version']) in req.specifier,(name,str(req))
        extras=set(req.extras)
        if name in active and extras<=active[name]:continue
        active.setdefault(name,set()).update(extras)
        for text in items[name]['metadata'].get('requires_dist',[]):
            child=Requirement(text)
            if child.marker is None or any(child.marker.evaluate({**resolution['environment'],'extra':extra}) for extra in active[name]|{''}):
                pending.append(child);checked.append(str(child))
    assert set(active)==set(items),(set(items)-set(active))
    result={'status':'PASS','independent_zipfile_packages':len(totals),'uncompressed_declared_bytes':sum(totals),
            'active_dependency_packages':len(active),'checked_active_requirements':len(checked),
            'torch':items['torch']['metadata']['version'],'torchvision':items['torchvision']['metadata']['version'],
            'source':'fixed pip resolution and official wheel directory metadata; no runtime imports or model calls',
            'installation_content_verified_here':False}
    (base/'CPU_COMPLETE_ENVIRONMENT_VERIFIED.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()
