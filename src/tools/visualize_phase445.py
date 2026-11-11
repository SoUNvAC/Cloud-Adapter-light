from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
R=Path('/home/scv/shared/phase445_20261106');P=Path('/home/scv/shared/phase444_retrain_20261103/source431/predictions/original_training')
rows=json.loads((R/'PER_IMAGE.json').read_text());kinds=[('Shadow to surface',2,0),('Shadow to cloud',2,1),('Surface to shadow',0,2),('Cloud to shadow',1,2)];selected=[]
for name,a,b in kinds:
 seen=set()
 for row in sorted(rows,key=lambda r:r['confusion'][a][b],reverse=True):
  if row['group'] in seen:continue
  seen.add(row['group']);selected.append((name,a,b,row))
  if len(seen)==2:break
font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';f=ImageFont.truetype(font,23);small=ImageFont.truetype(font,18);size=384;left=275;top=60;step=410
canvas=Image.new('RGB',(left+4*(size+12),top+8*step),'white');draw=ImageDraw.Draw(canvas);palette=np.array([[95,145,90],[235,235,235],[85,75,150]],np.uint8)
for j,name in enumerate(['Original RGB','Ground truth','Prediction','Selected error (red)']):draw.text((left+j*(size+12),15),name,font=f,fill='black')
manifest=[]
for i,(name,a,b,row) in enumerate(selected):
 rgb=np.array(Image.open(row['image']).convert('RGB'));raw=np.asarray(Image.open(row['mask']));y=raw.copy()
 for k,v in {0:0,1:1,2:1,3:2}.items():y[raw==k]=v
 with np.load(P/f"{row['index']:03d}.npz") as p:pred=p['pred']
 mask=(y==a)&(pred==b);overlay=(rgb.astype(float)*.55).astype(np.uint8);overlay[mask]=[255,45,40]
 for j,img in enumerate([rgb,palette[y],palette[pred],overlay]):canvas.paste(Image.fromarray(img).resize((size,size),Image.Resampling.NEAREST if j in [1,2] else Image.Resampling.LANCZOS),(left+j*(size+12),top+i*step))
 draw.text((12,top+i*step+12),name,font=f,fill='black');draw.text((12,top+i*step+52),f"index {row['index']}\n{row['group']}\n{int(mask.sum()):,} error pixels",font=small,fill='#444444')
 manifest.append(dict(error=name,index=row['index'],group=row['group'],pixels=int(mask.sum())))
canvas.save(R/'ERROR_EXAMPLES.png');canvas.thumbnail((1200,2000));canvas.save(R/'ERROR_EXAMPLES_preview.jpg',quality=93);(R/'EXAMPLE_SELECTION.json').write_text(json.dumps(dict(selection='Two largest pixel-error images per confusion direction, distinct source ROIs within each direction; diagnostic examples not prevalence estimate',rows=manifest),indent=2))
