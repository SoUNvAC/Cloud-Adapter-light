"""Exact observed RGB tiles, random sample; no per-image display enhancement."""
from pathlib import Path
import json,hashlib,random
import numpy as np
import mmcv
from PIL import Image,ImageDraw,ImageFont
R=Path('/home/scv/shared/phase444_retrain_20261103/appearance_montage');A=Path('/home/scv/shared/phase443_20261103')
def main():
    R.mkdir(exist_ok=True);rows=json.loads((A/'INPUTS.json').read_text());rng=random.Random(20261106)
    source=rng.sample([r for r in rows if r['domain']=='A'],24);target=rng.sample([r for r in rows if r['domain']=='B'],24)
    maps=dict(np.load(A/'ALIGNMENT.npz'));tile=512;gap=14;block=3*tile+2*gap;outer=44;between=62;top=150;rowh=554
    width=outer*2+block*3+between*2;height=top+rowh*8+76
    canvas=Image.new('RGB',(width,height),'white');draw=ImageDraw.Draw(canvas)
    fontpath=R/'msyh.ttc';title=ImageFont.truetype(str(fontpath),48);small=ImageFont.truetype(str(fontpath),24);caption=ImageFont.truetype(str(fontpath),27)
    titles=['源域原图 · CloudSEN','目标域原图 · Catalogue','目标域调色后 · Catalogue']
    for b,t in enumerate(titles):
        x=outer+b*(block+between);bbox=draw.textbbox((0,0),t,font=title);draw.text((x+(block-bbox[2])/2,36),t,fill='#17232f',font=title)
        sub='随机抽取24张源验证图' if b==0 else ('随机抽取24组目标图' if b==1 else '与中间逐一对应 · Phase443固定映射')
        box=draw.textbbox((0,0),sub,font=small);draw.text((x+(block-box[2])/2,100),sub,fill='#52606d',font=small)
    manifest=[]
    for b,selected in enumerate([source,target,target]):
        for i,row in enumerate(selected):
            p=Path(row['image']);assert hashlib.sha256(p.read_bytes()).hexdigest()==row['image_sha']
            if b==0:rgb=mmcv.imread(str(p))[...,::-1].copy()
            else:rgb=np.load(p)
            rgb=mmcv.imresize(rgb,(512,512),interpolation='bilinear').astype(np.float32)
            if b==2:
                mapped=np.empty_like(rgb)
                for c in range(3):mapped[...,c]=np.interp(rgb[...,c],maps['B'+str(c)+'_x'],maps['B'+str(c)+'_y'])
                rgb=mapped
            display=np.rint(np.clip(rgb,0,255)).astype(np.uint8)
            x=outer+b*(block+between)+(i%3)*(tile+gap);y=top+(i//3)*rowh
            canvas.paste(Image.fromarray(display),(x,y));draw.rectangle((x-1,y-1,x+tile,y+tile),outline='#d9dfe5',width=1)
            label=('S' if b==0 else 'T')+f'{i+1:02d}'
            draw.text((x+4,y+tile+5),label,fill='#344454',font=caption)
            manifest.append(dict(panel=b,slot=i+1,label=label,index=row['index'],group=row['group'],image=row['image'],image_sha=row['image_sha'],display_clipped_fraction=float(((rgb<0)|(rgb>255)).any(-1).mean())))
    footer='8行×3列／组｜固定随机种子20261106｜统一RGB 0–255显示，超范围仅在展示时截断；不做逐图拉伸'
    draw.text((outer,height-54),footer,fill='#52606d',font=small)
    assert [v['index'] for v in manifest[24:48]]==[v['index'] for v in manifest[48:]]
    canvas.save(R/'appearance_24x3_full.png',dpi=(300,300));preview=canvas.copy();preview.thumbnail((1800,1800));preview.save(R/'appearance_24x3_preview.jpg',quality=94)
    canvas.save(R/'appearance_24x3_full.jpg',quality=97,subsampling=0,dpi=(300,300))
    (R/'SAMPLES.json').write_text(json.dumps(dict(seed=20261106,source_population=535,target_population=32,source_display='Original, not A-mapped',alignment_sha=hashlib.sha256((A/'ALIGNMENT.npz').read_bytes()).hexdigest(),size=[width,height],rows=manifest),indent=2))
    print(json.dumps(dict(size=[width,height],tiles=72,source_unique=len({r['index'] for r in source}),target_unique=len({r['index'] for r in target}),target_pairs_exact=True)))
if __name__=='__main__':main()
