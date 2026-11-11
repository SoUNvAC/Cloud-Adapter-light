"""Recover only required original RGB into fresh Phase400, verify every old SHA."""
from phase400_common import *
import io,zipfile,copy
import numpy as np
def main():
 started=time.monotonic();assert not (R/'INPUT_LOCK.json').exists();assert sha(SOURCE)==EXPECTED_SOURCE;meta=js(R/'PREFLIGHT_READONLY.json');base=Path(meta['prepared_root']);manifest=base/'manifest.json';assert sha(manifest)==meta['manifest_SHA'];old=js(manifest);root=base.parent;rows=[];repairs=[]
 assert old['original_split_sha256']=='a431745e7a17bbd4a7b9e12ef3bf76767d4eb827911bd5cb5e632683c2177f7b'
 chosen=[r for r in old['rows'] if (r['split']=='fit' and r['is_representative']) or r['split']=='development_val'];assert len(chosen)==185
 with zipfile.ZipFile(root/'subscenes.zip') as archive:
  for row in chosen:
   row=copy.deepcopy(row);product=row['product'];rgb=base/row['rgb_path'];mask=base/row['mask_path'];valid=base/row['image_valid_path']
   for suffix,p in [('_mask.npy',mask),('_image_valid.npy',valid)]:assert sha(p)==row['prepared_sha256'][suffix],str(p)
   if not rgb.exists():
    rawbytes=archive.read('subscenes/'+product+'.npy');assert hashlib.sha256(rawbytes).hexdigest()==row['member_sha256']['image'];raw=np.load(io.BytesIO(rawbytes),allow_pickle=False);assert raw.shape==(1022,1022,13) and raw.dtype==np.float32;six=raw[...,[3,2,1,7,11,12]];iv=np.isfinite(six).all(-1)&(six!=0).all(-1);assert np.array_equal(iv,np.load(valid,allow_pickle=False));a=raw[...,[3,2,1]].copy()*np.float32(255);a[~iv]=0;rgb=R/'inputs/arrays'/(product+'_rgb.npy');rgb.parent.mkdir(parents=True,exist_ok=True);assert not rgb.exists();np.save(rgb,a);repairs.append({'product':product,'official_member_SHA':row['member_sha256']['image'],'prepared_RGB_SHA':sha(rgb),'original_missing':True,'restored_only_in_new_phase':True});del raw,a,six,iv,rawbytes
   assert sha(rgb)==row['prepared_sha256']['_rgb.npy'];y=np.load(mask,allow_pickle=False);counts=np.bincount(y[y!=255],minlength=3).tolist();assert counts==row['label_counts'] and int((y!=255).sum())==row['valid_pixels'];row.update(rgb_path=str(rgb),mask_path=str(mask),image_valid_path=str(valid));rows.append(row)
 save('PREPARED_MANIFEST.json',{**old,'rows':rows,'original_manifest_SHA':sha(manifest),'phase400_selection':'153 fit representatives, 32 fixed development; excludes 2 nonrepresentative fit products'})
 sources=[];source_root=Path('/home/scv/Cloud-Adapter-light/src/data/cloudsen12_high_l1c');from PIL import Image
 for image in sorted((source_root/'img_dir/val').glob('*.png')):
  mask=source_root/'ann_dir/val'/image.name;assert mask.is_file();a=np.asarray(Image.open(mask));assert set(np.unique(a))<=set([0,1,2,3,255]);counts=[int((a==0).sum()),int(((a==1)|(a==2)).sum()),int((a==3).sum())];sources.append({'name':image.name,'image':str(image),'mask':str(mask),'image_SHA':sha(image),'mask_SHA':sha(mask),'parent_counts':counts})
 assert len(sources)==535;fit=[r for r in rows if r['split']=='fit'];dev=[r for r in rows if r['split']=='development_val'];assert len(fit)==153 and len(dev)==32 and len({r['group_id'] for r in dev})==32
 save('RGB_RESTORATION.json',{'restored':repairs,'seconds':time.monotonic()-started,'no_download':True,'historical_files_unchanged':True});save('INPUT_LOCK.json',{'passed':True,'Source_SHA':sha(SOURCE),'original_manifest':str(manifest),'original_manifest_SHA':sha(manifest),'new_manifest_SHA':sha(R/'PREPARED_MANIFEST.json'),'original_split_SHA':old['original_split_sha256'],'fit_representatives':153,'development_groups':32,'development_shadow_groups':sum(r['label_counts'][2]>0 for r in dev),'source_val':sources,'source_val_root':str(source_root),'target_rows':rows,'class_order':['surface_visible','cloud','shadow'],'mask_policy':old['validity_policy'],'rgb_policy':old['rgb_policy'],'augmentation_seed':4000103,'data_sampler_seed':40001,'optimization_seed':40001,'no_source_train_or_test_pixels':True,'no_confirmation_pixels':True});print(json.dumps({'passed':True,'RGB_restored':len(repairs),'dev_shadow_groups':sum(r['label_counts'][2]>0 for r in dev),'source_val':len(sources)}))
if __name__=='__main__':main()
