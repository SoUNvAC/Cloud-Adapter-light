"""Prepare fixed ALCD-only blind visual review; no model or sealed-test access."""
import csv
import hashlib
import json
from pathlib import Path
import tarfile
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.windows import Window
from PIL import Image, ImageDraw


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    for p in (a.base, a.output):
        if not p.resolve().is_relative_to(Path('/home/scv/shared')):
            raise ValueError('Outside authorized shared data directory')
    a.output.mkdir(exist_ok=False)
    (a.output/'images').mkdir(); (a.output/'windows').mkdir()
    root = a.base/'imagery'
    manifest = json.loads((root/'product_manifest.json').read_text())
    status = json.loads((root/'imagery_download_status.json').read_text())
    geometry = json.loads((root/'geometry_audit.json').read_text())
    assert geometry['scenes_all_six_grids_match'] == 37
    archive = a.base/'SENTINEL_2_reference_cloud_masks_Baetens_Hagolle.tgz'
    assert sha(archive) == manifest['archive_sha256']
    differences = {s['key']: s for s in json.loads((root/'reference_disagreement_20261008.json').read_text())['scenes']}
    reference_path=root/'literature_CESBIO2024.tar.gz'
    assert sha(reference_path)=='441d8f6667c9059a8335aa14d2de0772fe5e0a7672aa1a28563f4742ed06516d'
    reference=tarfile.open(reference_path,'r:gz')
    geom = {s['key']: s for s in geometry['scenes']}
    samples = []; empty = []; skipped = []
    palette = np.array([[90,90,90],[90,90,90],[255,255,255],[255,255,255],[60,90,200],[90,180,90],[90,180,90],[90,180,90]], dtype=np.uint8)
    with tarfile.open(archive, 'r:gz') as tar:
        for scene_no, s in enumerate(manifest['scenes'], 1):
            key = s['key']; paths = {}
            for o in s['objects']:
                if o['name'].endswith('.jp2'):
                    entry = status['files'][o['name']]
                    p = root/entry['path']
                    assert sha(p) == entry['sha256']
                    paths[Path(o['name']).stem.split('_')[-1]] = p
            with MemoryFile(tar.extractfile(s['label_members'][0]).read()) as mem, mem.open() as lab:
                labels = lab.read(1); valid = np.isin(labels, [2,3,4,5,6,7])
                revised=None
                if key in differences:
                    with MemoryFile(reference.extractfile(differences[key]['reference_member']).read()) as rm, rm.open() as ri:
                        revised=ri.read(1)
                for band, p in paths.items():
                    with rasterio.open(p) as im:
                        arr = im.read(1); mask = im.read_masks(1)>0
                        for v in {0, *geom[key]['special_values'].values()}: mask &= arr != v
                        factor = int(60/im.res[0])
                        valid &= mask.reshape(1830,factor,1830,factor).all(axis=(1,3))
                        del arr, mask
                assert int(valid.sum()) == geom[key]['joint_valid_pixels_60m']
                points = {}
                for by in range(5):
                    for bx in range(5):
                        ys, xs = np.where(valid[by*366:(by+1)*366, bx*366:(bx+1)*366])
                        if not len(ys): empty.append([key,by,bx]); continue
                        candidates = ((hashlib.sha256(f'ALCD_REVIEW_V1|{key}|{int(y)+by*366}|{int(x)+bx*366}'.encode()).digest(),int(y)+by*366,int(x)+bx*366) for y,x in zip(ys,xs))
                        _, y, x = min(candidates)
                        points[(y,x)] = ['uniform']
                for direction, examples in differences.get(key, {}).get('review_examples', {}).items():
                    for p in examples:
                        y,x = p['row_60m'],p['col_60m']
                        if not valid[y,x]: skipped.append([key,y,x,direction]); continue
                        points.setdefault((y,x),[]).append(direction)
                for (y,x), groups in sorted(points.items()):
                    sid = f'R{len(samples)+1:04d}'
                    y0,y1=max(0,y-5),min(1830,y+6); x0,x1=max(0,x-5),min(1830,x+6)
                    arrays={}; stretches={}
                    for band,p in paths.items():
                        with rasterio.open(p) as im:
                            factor=int(60/im.res[0]); arr=im.read(1,window=Window(x0*factor,y0*factor,(x1-x0)*factor,(y1-y0)*factor))
                            arrays[band]=arr
                    np.savez_compressed(a.output/'windows'/f'{sid}.npz', **arrays, labels=labels[y0:y1,x0:x1], joint_valid=valid[y0:y1,x0:x1])
                    panels=[]
                    for name,bands in [('RGB',['B04','B03','B02']),('NIR SWIR',['B08','B11','B12'])]:
                        channels=[]
                        for b in bands:
                            ar=arrays[b]; good=(ar!=0)
                            for v in geom[key]['special_values'].values(): good &= ar!=v
                            lo,hi=np.percentile(ar[good],[2,98]) if good.any() else (0,1)
                            stretches[b]=[float(lo),float(hi)]
                            channels.append(np.uint8(np.clip((ar.astype(float)-lo)/max(hi-lo,1),0,1)*255))
                        size=max(z.shape[0] for z in channels)
                        channels=[np.asarray(Image.fromarray(z).resize((size,size),Image.Resampling.NEAREST)) for z in channels]
                        img=Image.fromarray(np.stack(channels,axis=-1)).resize((440,440),Image.Resampling.NEAREST)
                        draw=ImageDraw.Draw(img)
                        cx=(x-x0+.5)/(x1-x0)*440; cy=(y-y0+.5)/(y1-y0)*440
                        half=220/(x1-x0)
                        draw.rectangle((cx-half,cy-half,cx+half,cy+half),outline='red',width=2)
                        draw.text((5,5),name,fill='red'); panels.append(img)
                    blind=Image.new('RGB',(880,440));blind.paste(panels[0]);blind.paste(panels[1],(440,0));blind.save(a.output/'images'/f'{sid}.png')
                    labimg=Image.fromarray(palette[labels[y0:y1,x0:x1]]).resize((440,440),Image.Resampling.NEAREST)
                    for im in [labimg]:
                        ImageDraw.Draw(im).rectangle((cx-half,cy-half,cx+half,cy+half),outline='red',width=2)
                    combined=Image.new('RGB',(880,440));combined.paste(labimg)
                    revised_center=None
                    if revised is not None:
                        lut=np.full((256,3),90,dtype=np.uint8);lut[64]=[60,90,200];lut[128]=[90,180,90];lut[255]=[255,255,255]
                        rim=Image.fromarray(lut[revised[y0*3:y1*3,x0*3:x1*3]]).resize((440,440),Image.Resampling.NEAREST)
                        ImageDraw.Draw(rim).rectangle((cx-half,cy-half,cx+half,cy+half),outline='red',width=2)
                        combined.paste(rim,(440,0));revised_center=np.unique(revised[y*3:(y+1)*3,x*3:(x+1)*3]).tolist()
                    else: ImageDraw.Draw(combined).text((460,20),'No revised reference',fill='white')
                    combined.save(a.output/'images'/f'{sid}_label.png')
                    xx,yy=lab.xy(y,x)
                    samples.append(dict(id=sid,product=key,row=y,col=x,x=xx,y=yy,crs=str(lab.crs),groups=groups,original_class=int(labels[y,x]),revised_center=revised_center,window=[y0,y1,x0,x1],stretch=stretches,input_sha256={b:geom[key]['bands'][b]['sha256'] for b in paths}))
            print(f'scene {scene_no}/37: {len(samples)} samples',flush=True)
    fields=['id','judgement','reason','needs_clear_image_or_dem','after_reveal_note','reviewer','blind_locked']
    with (a.output/'review_results_template.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({'id':s['id']} for s in samples)
    (a.output/'sample_manifest.json').write_text(json.dumps(dict(samples=samples,empty_uniform_blocks=empty,skipped_invalid_difference_points=skipped,scope='ALCD_only_no_model'),ensure_ascii=False,indent=2),encoding='utf-8')
    # Local browser app keeps blind decisions before revealing labels; export is the deliverable.
    html='''<!doctype html><meta charset="utf-8"><title>ALCD 标签复核</title><style>body{font:18px sans-serif;max-width:1000px;margin:20px auto}img{max-width:100%}button,select,input{font:inherit;margin:6px}textarea{width:90%;height:70px}</style><h1>ALCD 标签复核</h1><p>看红框内中心60m格。先填写判断和理由，再揭示标签。暗色不一定是云影；分不清请选无法判定。左RGB，右B08/B11/B12假彩色。颜色拉伸仅辅助观察。</p><p>结果自动保存在当前浏览器；结束或中途务必导出CSV交回。保存HTML并不能保存填写结果。</p><button onclick="prev()">上一处</button><button onclick="next()">保存并下一处</button><button onclick="exportCsv()">导出填写结果CSV</button><input id="jump" type="number" min="1" style="width:90px"><button onclick="go()">跳转</button><div id="info"></div><img id="blind"><br><select id="judge"><option value="">请选择</option><option>Cloud</option><option>Shadow</option><option>Surface</option><option>混合</option><option>无法判定</option></select><p>Cloud=云；Shadow=云影；Surface=地表（包括水/雪）。</p><textarea id="reason" placeholder="判读理由；例如暗区可能是水，缺少晴空图无法确定"></textarea><br><label><input type="checkbox" id="need">需要晴空影像或DEM</label><br><button onclick="reveal()">保存判断并揭示原标签</button><div id="labels" hidden><p id="truth"></p><img id="label"><p>左：原参考；右：公开修订参考（若有）。白=云，蓝=云影，绿=地表，灰=无效。参考图不等于真值。</p><textarea id="note" placeholder="揭示后发现的冲突或补充；保留上面的独立判断"></textarea></div><script>const samples=DATA;const storeKey='phase65-alcd-review-v1';let data=JSON.parse(localStorage.getItem(storeKey)||'{}'),i=0;function save(){let old=data[samples[i].id]||{};data[samples[i].id]={judgement:old.blind_locked?old.judgement:judge.value,reason:old.blind_locked?old.reason:reason.value,needs_clear_image_or_dem:old.blind_locked?old.needs_clear_image_or_dem:(need.checked?'yes':'no'),blind_locked:old.blind_locked||false,after_reveal_note:note.value,reviewer:''};localStorage.setItem(storeKey,JSON.stringify(data))}function show(){let s=samples[i],d=data[s.id]||{};info.textContent=s.id+' / '+samples.length+' — '+s.product+' — row '+s.row+', col '+s.col;blind.src='images/'+s.id+'.png';label.src='images/'+s.id+'_label.png';judge.value=d.judgement||'';reason.value=d.reason||'';need.checked=d.needs_clear_image_or_dem==='yes';note.value=d.after_reveal_note||'';labels.hidden=true;judge.disabled=!!d.blind_locked;reason.readOnly=!!d.blind_locked;need.disabled=!!d.blind_locked;jump.value=i+1}function next(){save();i=Math.min(i+1,samples.length-1);show()}function prev(){save();i=Math.max(0,i-1);show()}function go(){save();i=Math.max(0,Math.min(samples.length-1,Number(jump.value)-1));show()}function reveal(){if(!judge.value||!reason.value.trim()){alert('先填写判断与理由；无法判断也是有效答案。');return}save();data[samples[i].id].blind_locked=true;localStorage.setItem(storeKey,JSON.stringify(data));judge.disabled=true;reason.readOnly=true;need.disabled=true;labels.hidden=false;truth.textContent='原参考中心类号：'+samples[i].original_class+'（2/3云，4云影，5/6/7地表）'}function exportCsv(){save();let fields=['id','judgement','reason','needs_clear_image_or_dem','after_reveal_note','reviewer','blind_locked'];let q=v=>'"'+String(v||'').replaceAll('"','""')+'"';let lines=[fields.join(',')];samples.forEach(s=>{let d=data[s.id]||{};lines.push(fields.map(f=>q(f==='id'?s.id:d[f])).join(','))});let a=document.createElement('a');a.href=URL.createObjectURL(new Blob(['\\ufeff'+lines.join('\\r\\n')],{type:'text/csv;charset=utf-8'}));a.download='review_results.csv';a.click()}show();</script>'''
    (a.output/'index.html').write_text(html.replace('DATA',json.dumps(samples,ensure_ascii=False)),encoding='utf-8')
    (a.output/'README.txt').write_text('打开index.html。先看照片填写判断和理由，再揭示标签。中途可导出保存。交回review_results.csv即可；不用交回图像或原始影像。不要改id。不确定请选择无法判定。\n诊断样本不可用于总体错误率。普通抽样与版本差异点组别保存在sample_manifest.json，盲看时不显示。\n',encoding='utf-8')
    files={str(p.relative_to(a.output)):sha(p) for p in a.output.rglob('*') if p.is_file()}
    (a.output/'SHA256.json').write_text(json.dumps(files,indent=2),encoding='utf-8')
    print(json.dumps({'samples':len(samples),'empty_blocks':len(empty),'skipped_invalid_difference_points':len(skipped)}),flush=True)


if __name__=='__main__': main()
