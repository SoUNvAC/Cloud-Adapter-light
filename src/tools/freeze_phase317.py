"""Freeze executable GPU code and reference file locations without opening references."""
from phase317_common import *
def main():
 assert not (ROOT/'GPU_CODE_LOCK.json').exists() and (ROOT/'EXECUTION_LOCK.json').exists();resource_check()
 raw={}
 for row in inputs('L8')['val']:
  p=Path('/home/scv/shared/data/l8_biome_raw/l8biome')/row['biome']/row['scene']/(row['scene']+'_fixedmask.TIF');assert p.is_file(),p
  raw[row['scene']]=dict(path=str(p),bytes=p.stat().st_size,note='stat only; open afterallfiveendpointssealed; restorefill collapsedbyoriginalPNGconverter')
 write(ROOT/'REFERENCE_METADATA.json',dict(l8_raw=raw,L8_native_codes={'0':'fill','64':'shadow','128':'clear','192':'thin','255':'cloud'},PNG_codes={'0':'clear_or_collapsedfill','1':'shadow','2':'thin','3':'cloud'},rule='global coordinate orientation must reproduce originalPNG exactly under oldordinalmap before excluding rawfill; ambiguity only allowed if bothyieldidentical reference',pixels_read=False))
 paths=[p for p in (R/'src/tools').glob('*phase317*.py')]
 write(ROOT/'GPU_CODE_LOCK.json',{p.relative_to(R).as_posix():sha(p) for p in paths})
 for p in paths:shutil.copyfile(p,ROOT/'code'/p.name)
 write(ROOT/'FROZEN_CONFIG.json',dict(inputlock_sha=sha(ROOT/'EXECUTION_LOCK.json'),reference_metadata_sha=sha(ROOT/'REFERENCE_METADATA.json'),code_lock_sha=sha(ROOT/'GPU_CODE_LOCK.json'),strategy='CB equal means of presentacceptedclasses, PL acceptedpixelmean',initial='fresh30201SourceLoRA compared to304 initial; no oldtargethead/optimizer',label_read_barrier='allfiveENDPOINT_LOCK and all evaluations sealed, independent CPUonly',new_forwards_max=11000,gpu_seconds_max=43200,maxpersistent=16000000000))
 print('Phase317 GPU protocol frozen; no reference pixels read')
if __name__=='__main__':main()
