"""Derive independent CPU fit/mask audit and bounded head-only collection."""
from pathlib import Path
import ast
R=Path(__file__).resolve().parents[2]
def main():
 t=(R/'src/tools/collect_phase217.py').read_text(encoding='utf-8').replace('phase217','phase222').replace('20261023','20261024').replace('48893','99190').replace('48894','99191').replace('<14400','<21600').replace('[2,2,0,0,0,1,0]','[0,2,1,1]').replace('valid=v<7','valid=v<4').replace('and len(mask_cache)==60','and 0<len(mask_cache)<=1560')
 # Keep the old crop and flip verification; additionally tie mask identities to input lock.
 t=t.replace("for row in schedule:\n   p=row['mask_path']", "locked={v['name']:v for v in js(ROOT/'L8_TRAIN_INPUT_LOCK.json')}\n  for row in schedule:\n   p=row['mask_path'];record=locked[Path(p).name];assert sha(Path(p))==record['mask_sha256'] and sha(Path(row['image_path']))==record['image_sha256']")
 # Hash each unique image once, rather than hashing4000 uses repeatedly.
 t=t.replace("schedules={};models={}","schedules={};models={};checked_images=set()")
 t=t.replace("assert sha(Path(p))==record['mask_sha256'] and sha(Path(row['image_path']))==record['image_sha256']","\n   if p not in checked_images:\n    assert sha(Path(p))==record['mask_sha256'] and sha(Path(row['image_path']))==record['image_sha256'];checked_images.add(p)")
 t=t.replace("names=[p for p in ROOT.rglob('*')", "endpoint_names={f'seed{s}/heads/{a}_iter4000.pth' for s in [21701,21702] for a in ['C','D']}\n names=[p for p in ROOT.rglob('*')")
 t=t.replace("and p.suffix!='.pth'", "and (p.suffix!='.pth' or str(p.relative_to(ROOT)) in endpoint_names)")
 t=t.replace("with tarfile.open(archive,'w:gz') as t:","partial=archive.with_name(archive.name+'.partial');assert not partial.exists()\n with tarfile.open(partial,'w:gz') as t:")
 t=t.replace(" print(json.dumps({'archive'", " partial.replace(archive)\n print(json.dumps({'archive'")
 t=t.replace("'all8000_target_parent_crop_flip_masks_independently_checked':True", "'all8000_target_parent_crop_flip_masks_independently_checked':True,'mapping_L8':[0,2,1,1],'unique_training_masks_checked':len(mask_cache),'L8_input_preflight':js(ROOT/'INPUT_PREFLIGHT_ACCEPTED.json')")
 ast.parse(t);(R/'src/tools/collect_phase222.py').write_text(t,encoding='utf-8')
if __name__=='__main__':main()
