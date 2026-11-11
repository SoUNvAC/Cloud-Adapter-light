"""Independent label LUT/nearest-grid denominators, no model calls."""
from pathlib import Path
import csv,hashlib,json,time
import numpy as np
from PIL import Image
ROOT=Path('/home/scv/shared/phase213_20261021_run04');OUT=ROOT/'evaluation';SRC=Path('/home/scv/Cloud-Adapter-light/src')
assert (ROOT/'EXIT_CODE').read_text().strip()=='0'
assert (OUT/'FINAL_ADMISSION_LOCK.json').exists()
receipt=OUT/'REFERENCE_CPU_ACCEPTANCE.json'
if receipt.exists():print(receipt.read_text());raise SystemExit(0)
start=time.monotonic();counts={};classes={};ignore={}
for pool,mapping,manifest in [('source',[0,1,1,2],None),('l8',[0,2,1,1],SRC/'work_dirs/phase45_protocol_audit/scene_disjoint_manifest.csv'),('sparcs',[2,2,0,0,0,1,0],SRC/'work_dirs/phase64_sparcs_protocol_verified_v2/scene_disjoint_manifest.csv')]:
    raw=json.loads((OUT/f'{pool}_a0.json').read_text());expected={r['name']:r for r in raw['per_input']}
    if manifest:
        files={r['name']:Path(r['mask_path']) if Path(r['mask_path']).is_absolute() else SRC/r['mask_path'] for r in csv.DictReader(manifest.open()) if r['new_split']=='target_test'}
    else:
        source=SRC/'data/cloudsen12_high_l1c/ann_dir/val'
        if not source.is_dir():source=SRC.parent/'data/cloudsen12_high_l1c/ann_dir/val'
        files={p.name:p for p in source.iterdir() if p.is_file()}
    assert set(files)==set(expected);total=np.zeros(3,dtype=np.int64);ignored=0;seen=set()
    for name,p in sorted(files.items()):
        with Image.open(p) as im:label=np.array(im)
        if label.ndim==3:assert label.shape[2]==1;label=label[:,:,0]
        h,w=label.shape
        # Explicit floor-coordinate indexing reproduces locked OpenCV INTER_NEAREST.
        yy=(np.arange(512,dtype=np.int64)*h)//512;xx=(np.arange(512,dtype=np.int64)*w)//512
        label=label[np.ix_(yy,xx)];seen.update(int(x) for x in np.unique(label));ref=np.zeros(3,dtype=np.int64)
        for code,parent in enumerate(mapping):ref[parent]+=int(np.count_nonzero(label==code))
        missing=512*512-int(ref.sum());assert np.array_equal(ref,np.array(expected[name]['confusion']).sum(axis=1))
        assert missing==expected[name]['ignore_pixels'];total+=ref;ignored+=missing
    counts[pool]={'inputs':len(files),'reference_pixels_per_parent':total.tolist(),'ignore_pixels':ignored};classes[pool]=sorted(seen)
r={'status':'passed','independent_numpy_floor_nearest_grid_and_explicit_label_counts':True,'mask_palette_indices_preserved':True,'all_input_reference_denominators_match':True,'counts':counts,'observed_native_class_codes':classes,'class_mapping':{'source':[0,1,1,2],'l8':[0,2,1,1],'sparcs':[2,2,0,0,0,1,0]},'elapsed_seconds':time.monotonic()-start,'new_model_calls':0,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest() if '__file__' in globals() and Path(__file__).is_file() else 'stdin sealed locally'}
receipt.write_text(json.dumps(r,indent=2));print(json.dumps(r))
