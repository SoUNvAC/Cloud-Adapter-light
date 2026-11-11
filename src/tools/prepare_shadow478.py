from pathlib import Path
T=Path(__file__).parent
def convert(s):return s.replace('477','478').replace('phase478_20261107','phase478_20261108')
s=convert((T/'train_shadow477.py').read_text())
s=s.replace("assert ARM=='A'","assert ARM in ['A','B']\nPATHS=('backbone.refinement.','backbone.local.','backbone.output_residual.') if ARM=='A' else ('backbone.spatial.','backbone.interactions.')\nALLOWED=('decode_head.',)+PATHS")
s=s.replace("n.startswith(('decode_head.',))","n.startswith(ALLOWED)")
s=s.replace("assert sum(v for k,v in changed.items() if k not in ['decode_head.'])==0","assert all(not v for k,v in changed.items() if k not in ALLOWED)\n   assert all(changed[k]>0 for k in PATHS)")
s=s.replace('Only original decoder trainable; original backbone,450 branch and BN fixed','Decoder and selected original adaptation path trainable; VFM,450 branch and BN fixed')
s=s.replace('new_module=None,uniform_pixel_CE=True','new_module=None,trainable_path=list(PATHS),uniform_pixel_CE=True')
(T/'train_shadow478.py').write_bytes(s.encode())
for n in ['infer_shadow477.py','verify_shadow477.py','boundary_shadow477.py','collect_shadow477.py','verify_archive477.py']:
 (T/n.replace('477','478')).write_bytes(convert((T/n).read_text()).encode())
