"""Once-only bounded discovery, before the audit: existing metadata, no HTTP/pixels."""
from datetime import datetime,timezone,timedelta
from collections import Counter
import re
import subprocess
from phase316_common import *
def main():
    assert sys.platform=='linux' and not ROOT.exists()
    assert datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d')=='20261026'
    ROOT.mkdir();write(ROOT/'START.json',dict(unix=time.time(),beijing=datetime.now(timezone(timedelta(hours=8))).isoformat(),git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()))
    resource_check();(ROOT/'code').mkdir();(ROOT/'evidence').mkdir()
    for p in sorted(E.iterdir()):shutil.copyfile(p,ROOT/'evidence'/p.name)
    for p in (R/'src/tools').glob('*phase316*.py'):shutil.copyfile(p,ROOT/'code'/p.name)
    paths=[OLD/'inputs/history_positions_only.json',OLD/'inputs/candidate_products_metadata.json',R/'src/work_dirs/phase65a/source_mapping/mapping_audit.json',R/'src/work_dirs/phase64_source_parent/seed64/phase64_source_parent_rgb.py',Path('/home/scv/shared/data/phase65_source_metadata/source_metadata_manifest.json')]
    report=[]
    for p in paths:
        if not p.exists():report.append(dict(path=str(p),exists=False));continue
        item=dict(path=str(p),exists=True,bytes=p.stat().st_size,sha256=sha(p))
        if p.suffix=='.json':
            d=load(p);item['schema']=list(d) if isinstance(d,dict) else ['list',len(d)];item['first']=str(d[0] if isinstance(d,list) else {k:(v[:1] if isinstance(v,list) else str(v)[:900]) for k,v in d.items() if k not in ['mappings','records']})[:3500]
        report.append(item)
    patches=read(OLD/'inputs/kappa_patch_positions.csv');members=load(E/'kappaset_archive_index.json');nc=[r for r in members if r['path'].endswith('.nc')]
    products=[r for r in load(OLD/'inputs/candidate_products_metadata.json') if r['source']=='KappaSet7100327']
    print(json.dumps(dict(files=len(patches),keys=len({p['record_key'] for p in patches}),splits=dict(Counter(p['archive_path'].split('/')[1] for p in patches)),unique_tile_day=len({(p['tile'],p['sensing_utc_key'][:8]) for p in patches}),nc=len(nc),source_probe=report,geo=[r for r in products if r.get('geo')]),ensure_ascii=False))
    write(ROOT/'DISCOVERY.json',dict(metadata=report,new_requests=0,resources=resource_check()))
    print('Phase316 discovery only: no pixels, no models, no HTTP')
if __name__=='__main__':main()
