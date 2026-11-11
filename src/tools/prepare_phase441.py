"""Lock native-label patch fractions and ROI-level sampling before feature extraction."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
R=Path('/home/scv/shared/phase441_20261103')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,x):(R/n).write_text(json.dumps(x,indent=2,allow_nan=False))
def area_weights(size):
    edges=np.linspace(0,size,33)
    return np.maximum(0,np.minimum(edges[1:,None],np.arange(size)[None,:]+1)-np.maximum(edges[:-1,None],np.arange(size)[None,:]))/(size/32)
def main():
    lock=json.loads(Path('/home/scv/shared/phase400_20261031/INPUT_LOCK.json').read_text())
    bindings={r['png_name']:r for r in csv.DictReader((R/'SOURCE_ROI_BINDINGS.csv').open())}
    assert len(bindings)==535 and len({v['roi_id'] for v in bindings.values()})==107
    rows=[];candidates={};coverage=[];pi={d:np.zeros(3,np.int64) for d in ['A','B']}
    native=[]
    for domain,items in [('A',lock['source_val']),('B',[x for x in lock['target_rows'] if x['split']=='development_val'])]:
        for v in items:
            if domain=='A':
                image,mask=v['image'],v['mask'];assert sha(image)==v['image_SHA']==bindings[v['name']]['image_sha256'];assert sha(mask)==v['mask_SHA']
                raw=np.asarray(Image.open(mask));truth=raw.copy()
                for old,new in [(0,0),(1,1),(2,1),(3,2)]:truth[raw==old]=new
                group='A:'+bindings[v['name']]['roi_id'];name=v['name'];expected=v['parent_counts']
            else:
                image,mask=v['rgb_path'],v['mask_path'];assert sha(image)==v['prepared_sha256']['_rgb.npy'];assert sha(mask)==v['prepared_sha256']['_mask.npy']
                truth=np.load(mask);group='B:'+v['group_id'];name=v['product'];expected=v['label_counts']
            count=np.bincount(truth[truth!=255],minlength=3);assert count.tolist()==expected;pi[domain]+=count
            wy,wx=area_weights(truth.shape[0]),area_weights(truth.shape[1])
            fractions=np.stack([wy@(truth==c).astype(float)@wx.T for c in range(3)],-1).reshape(-1,3)
            valid=fractions.sum(1);proportions=fractions/np.maximum(valid[:,None],1e-20)
            label=proportions.argmax(1);purity=proportions.max(1)
            index=len(rows);rows.append(dict(index=index,domain=domain,group=group,name=name,image=image,mask=mask,
                image_sha=sha(image),mask_sha=sha(mask),shape=list(truth.shape),counts=count.tolist(),selected=[]))
            native.append(dict(fractions=proportions,valid=valid))
            for c in range(3):
                pure=(valid>=.9)&(label==c)&(purity>=.9)
                mixed=(valid>=.9)&(label==c)&(purity<.9)
                coverage.append(dict(index=index,domain=domain,group=group,category=c,raw_pixels=int(count[c]),
                    class_patch_mass=float(fractions[:,c].sum()),pure_tokens=int(pure.sum()),mixed_tokens=int(mixed.sum()),
                    retained_class_mass=float(fractions[pure,c].sum()),valid_tokens=int((valid>=.9).sum())))
                for kind,keep in [('pure',pure),('mixed',mixed)]:
                    candidates.setdefault((group,c,kind),[]).extend((index,int(t)) for t in np.flatnonzero(keep))
    rng=np.random.default_rng(44101);sampling=[]
    for (group,c,kind),items in sorted(candidates.items()):
        chosen=rng.choice(len(items),size=min(128,len(items)),replace=False)
        for j in sorted(chosen):
            index,token=items[j];entry=dict(token=token,category=c,kind=kind,fractions=native[index]['fractions'][token].tolist(),valid_fraction=float(native[index]['valid'][token]))
            rows[index]['selected'].append(entry)
        sampling.append(dict(group=group,category=c,kind=kind,available=len(items),selected=len(chosen)))
    save('INPUTS.json',rows);save('COVERAGE.json',coverage);save('SAMPLING.json',sampling)
    save('PROTOCOL.json',dict(seed=44101,source_images=535,source_ROIs=107,target_groups=32,grid=[32,32],
        purity=.9,min_valid_fraction=.9,max_tokens_per_scene_class_kind=128,
        source_group_binding_sha=sha(R/'SOURCE_ROI_BINDINGS.csv'),
        mask_alignment='Exact fractional area overlap of native label pixels with 32x32 resized-image patch footprints; no nearest-label purity proxy',
        pi_raw={d:(x/x.sum()).tolist() for d,x in pi.items()},raw_counts={d:x.tolist() for d,x in pi.items()},
        mixed='Dominant-class patches with purity <0.9, separately sampled and reported; ties use lowest class index',
        source_BN='Existing Phase434 source_fit BN, label-free target-fit calibration; zero target-label supervision, not no-target-image exposure',
        pca='Per model and stage, shared pre/attention/post 64D, domain/scene/location balanced, not whitened; CI conditional on this descriptive PCA',
        cv='Class-conditional ROI-grouped balanced folds; PCA shared across stage positions, and scaling fitted only within training fold',
        scene_weight='Each represented ROI/group equally weighted within domain and class; source ROI spans five images',
        bootstrap=256,null_repeats=100,SW_directions=64,SW_hist_bins=512,
        data_caveat='Source existing uint8 RGB conversion versus target linear reflectance RGB; measures actual pipeline domain difference; ROI grouping is not geographic independence certification'))
    print(json.dumps(dict(passed=True,images=len(rows),selected=sum(len(x['selected']) for x in rows),raw_pi={d:(x/x.sum()).tolist() for d,x in pi.items()})))
if __name__=='__main__':main()
