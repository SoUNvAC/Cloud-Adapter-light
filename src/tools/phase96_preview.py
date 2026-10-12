"""Lossless PNG previews using only numpy and Python stdlib; not model inputs."""
import struct,zlib
import numpy as np


def png(path,a):
    a=np.asarray(a,np.uint8); assert a.ndim==3 and a.shape[2]==3
    h,w=a.shape[:2]
    def chunk(tag,b):return struct.pack('>I',len(b))+tag+b+struct.pack('>I',zlib.crc32(tag+b)&0xffffffff)
    raw=b''.join(b'\0'+a[i].tobytes() for i in range(h))
    data=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',w,h,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw,6))+chunk(b'IEND',b'')
    with path.open('xb') as f:f.write(data)


def heat(a):
    x=np.clip(np.nan_to_num(a),0,1)
    return (np.stack([x,.25+.5*x,1-x],axis=-1)*255).astype(np.uint8)


def create(root,raw,source,g7,ss,truth,fp20,fp60):
    root.mkdir(); rgb=np.power(np.clip(raw[...,[3,2,1]]/.4,0,1),1/2.2)
    label=np.zeros((*truth.shape,3),np.uint8)
    for c,color in [(0,[65,170,100]),(1,[245,245,245]),(2,[130,70,180]),(255,[0,0,0])]:label[truth==c]=color
    arrays={'rgb20':(255*rgb).astype(np.uint8),'source20':heat(source),'g7_20':heat(g7),'sourceS20':heat(ss),'reference20':label}
    for name,mask in fp20.items(): arrays['false_positive20_'+name]=np.where(mask[...,None],np.array([255,40,40],np.uint8),np.array([15,15,15],np.uint8))
    for name,mask in fp60.items():arrays['false_positive60_'+name]=np.where(mask[...,None],np.array([255,40,40],np.uint8),np.array([15,15,15],np.uint8))
    for name,a in arrays.items():png(root/(name+'.png'),a)
    captions={'rgb20':'RGB20m: B04/B03/B02, display TOA 0..0.4, gamma 1/2.2 only; scoring unchanged',
        'source20':'Source shadow softmax proxy 0..1; NOT calibrated correctness probability',
        'g7_20':'G7 Shadow probability 0..1','sourceS20':'SourceS sigmoid probability 0..1',
        'reference20':'Reference: Surface green; Cloud white; Shadow purple; invalid black'}
    parts=['<!doctype html><meta charset="utf-8"><title>Phase96 T31TGF fixed previews</title>',
        '<style>body{font:16px sans-serif;background:#eee;margin:24px}.grid{display:grid;grid-template-columns:repeat(2,minmax(300px,1fr));gap:18px}figure{margin:0;background:white;padding:12px}img{width:100%;image-rendering:pixelated}a{color:#124}</style>',
        '<h1>T31TGF: same-position native previews</h1><p>All full-scene20m panels share the1022x1022 grid.60m masks use top-left1020 crop,340x340 blocks; same geographic extent except removed right/bottom2pixels. Click any image for native resolution. These are reference disagreements, not human-confirmed label errors.</p>',
        '<p>Probability colors:0=blue,1=yellow. FP masks:red=false positive under frozen threshold, dark=other. RGB display saturation does not change any diagnostic input. T20_full is primary; other masks are sensitivity only.</p><div class="grid">']
    for name in arrays:
        caption=captions.get(name,name+' (red=reference nonshadow predicted Shadow)')
        parts.append(f'<figure><figcaption>{caption}</figcaption><a href="{name}.png"><img src="{name}.png"></a></figure>')
    parts.append('</div>');(root/'index.html').write_text('\n'.join(parts),encoding='utf-8')
    return dict(panels=list(arrays),rgb_display_only='clip TOA/0.4, gamma1/2.2; no numerical input clipping',native20_shape=[1022,1022],native60_shape=[340,340],probability_color='blue0-yellow1',no_manual_semantic_confirmation=True)
