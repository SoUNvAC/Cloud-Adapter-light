"""Small evidence delivery from fixed predictions and presealed candidate points."""
import base64,csv,html,io
from phase200_common import *
def main():
    import numpy as np
    from PIL import Image,ImageDraw
    state=load(ROOT/'infer/STATUS.json');assert state['status']=='completed'
    sealed=load(ROOT/'POINTS_SEALED.json');assert sha(ROOT/'candidate_points.csv')==sealed['points_sha256']
    plan=load(LOCK);inp=load(ROOT/'INPUT_RECEIPT.json');assets=load(ROOT/'ASSET_VERIFICATION.json')
    with (ROOT/'candidate_points.csv').open(encoding='utf-8-sig',newline='') as f:points=list(csv.DictReader(f))
    out=ROOT/'delivery';out.mkdir(exist_ok=False)
    win=inp['display_window'];r,c,h,w=[win[k] for k in ['row','col','height','width']]
    rgb=Image.open(ROOT/'true_color.png').convert('RGB');palette=np.array([[0,0,0],[255,50,50],[0,200,255],[255,230,0]],dtype='uint8')
    imgs={'true_color':rgb.copy()};predictions={}
    for version in ['V1','V4']:
        s=state['versions'][version];assert sha(ROOT/(version+'_shadow.npy'))==s['shadow_sha256'] and sha(ROOT/(version+'_classes.npy'))==s['classes_sha256']
        classes=np.load(ROOT/(version+'_classes.npy'),mmap_mode='r');scores=np.load(ROOT/(version+'_shadow.npy'),mmap_mode='r')
        cls=np.array(classes[r:r+h,c:c+w]);base=np.array(rgb);mask=(cls>0)&(cls<4)
        overlay=base.copy();overlay[mask]=(.4*base[mask]+.6*palette[cls[mask]]).astype('uint8');imgs[version]=Image.fromarray(overlay)
        predictions[version]={}
        for x in points:
            rr,cc=int(x['row20']),int(x['col20']);cl=int(classes[rr,cc]);score=float(scores[rr,cc]);assert 0<=cl<=3 and np.isfinite(score)
            predictions[version][x['point_id']]=(cl,score);x[version+'_class']=['Clear','ThickCloud','ThinCloud','Shadow'][cl];x[version+'_shadow_score']=repr(score)
        del classes,scores
    for name,image in imgs.items():
        draw=ImageDraw.Draw(image)
        for x in points:
            px,py=int(x['col20'])-c,int(x['row20'])-r;draw.ellipse((px-4,py-4,px+4,py+4),outline='white',width=1);draw.text((px+5,py-6),x['point_id'],fill='white',stroke_width=1,stroke_fill='black')
        image.save(out/(name+'.png'))
    with (out/'points.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(points[0]));writer.writeheader();writer.writerows(points)
    def count(group,version):
        p=[x for x in points if x['candidate']==group and x['certainty']=='clear'];return f"{sum(x[version+'_class']=='Shadow' for x in p)}/{len(p)}"
    changed={}
    for group in ['shadow','shore','water','cloud']:
        changed[group]=dict(new_shadow=[x['point_id'] for x in points if x['candidate']==group and x['certainty']=='clear' and x['V1_class']!='Shadow' and x['V4_class']=='Shadow'],
            removed_shadow=[x['point_id'] for x in points if x['candidate']==group and x['certainty']=='clear' and x['V1_class']=='Shadow' and x['V4_class']!='Shadow'])
    write(out/'CANDIDATE_COUNTS.json',dict(counts={g:{v:count(g,v) for v in ['V1','V4']} for g in changed},changes=changed,not_independent_truth=True))
    lock=dict(execution_plan=plan,points_sealed=sealed,input=inp,models={k:v['bindings'] for k,v in state['versions'].items()},input_identical=True,threshold_search=False,decision='official default raw argmax')
    write(out/'input_model_lock.json',lock);write(out/'RESOURCE_RECEIPT.json',dict(acquire=load(ROOT/'acquire/STATUS.json'),infer=state,actual_download_bytes=assets['actual_download_bytes'],paper_historical_settings_unknown=True,training_data_overlap='unknown',not_independent_validation=True))
    headers=list(points[0]);table='<table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x[k]))+'</td>' for k in headers)+'</tr>' for x in points)+'</table>'
    figures=''.join('<figure><figcaption>'+name+'</figcaption><img src="data:image/png;base64,'+base64.b64encode((out/(name+'.png')).read_bytes()).decode()+'"></figure>' for name in imgs)
    page='<!doctype html><meta charset="utf-8"><title>Phase200 fixed lake case</title><style>body{font:15px sans-serif;margin:20px}section{display:flex;gap:12px}figure{margin:0;flex:1}img{width:100%;min-width:400px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:5px}figcaption{font-weight:bold}</style><h1>Phase200: Lake Cascade 2023-09-08</h1><p>Same native 20m DN input; 1000/300, batch1, float32, full 2-model ensembles. Clear=uncolored; ThickCloud=red; ThinCloud=cyan; Shadow=yellow. Candidate points were AI-selected from RGB before outputs; not independent ground truth. Confidence is not correctness probability.</p><section>'+figures+'</section><h2>Fixed candidate points (20m cells)</h2>'+table
    (out/'evidence.html').write_text(page,encoding='utf8')
    n_mixed=sum(x['certainty']=='mixed' for x in points);n_uncertain=sum(x['certainty']=='uncertain' for x in points);missing=sum(sealed['missing'].values())
    signs={v:any(x['candidate']=='shore' and x['certainty']=='clear' and x[v+'_class']=='Shadow' for x in points) for v in ['V1','V4']}
    rows=[('完整产品ID、日期、tile、处理级别',plan['product_id']+' / L1C','同一输入','匹配2023-09-08 / 11TNK'),
        ('官方代码／模型身份及SHA','V1完整2模型','V4完整2模型','[冻结身份](input_model_lock.json)'),('输入与有效范围一致','是','是','同一image20/valid20 SHA；整tile推理、图3湖区显示')]
    for group,label in [('shadow','候选云影'),('shore','候选清晰岸边'),('water','候选清晰湖水'),('cloud','候选云')]:rows.append((label+'判为Shadow的点数',count(group,'V1'),count(group,'V4'),'仅计AI明确候选，不是正式召回或误报率'))
    rows += [('混合／不确定／缺额点数',f'{n_mixed}/{n_uncertain}/{missing}','同一批点','保留原判读，不纳入明确候选分母'),
        ('原文岸边疑似误报现象','有线索' if signs['V1'] else '未在明确候选点观察到','有线索' if signs['V4'] else '未在明确候选点观察到','[同范围证据](evidence.html)，无独立真值不能确诊误报'),
        ('V4相对V1新增或减少的问题','—',json.dumps(changed,ensure_ascii=False),'new_shadow/removed_shadow按候选类列全；候选影减少可能是漏影线索'),
        ('原文权重、输入、设置匹配程度','原文未知','历史版本不同','本轮是案例核验，非严格复现'),('本例与模型训练数据交集','未知','未知','不称独立验证'),
        ('实际下载量、模型调用、耗时、峰值资源',f"新增共{assets['actual_download_bytes']} bytes；V1权重复用",f"总调用{state['model_calls']}；GPU阶段{state['gpu_phase_seconds']:.1f}s",'[完整资源回执](RESOURCE_RECEIPT.json)，含各版本耗时／峰值；无配对速度结论'),
        ('工程验收及最终状态','完成','完成','输入SHA、有效范围、类别、有限值、调用上限通过')]
    result='| 检查项 | V1 | V4 | 结论／证据链接 |\n|---|---|---|---|\n'+'\n'.join('| '+' | '.join(str(v).replace('|','/').replace('\n',' ') for v in row)+' |' for row in rows)
    conclusion='有线索' if signs['V4'] else '暂未在明确候选点观察到，仍证据不足以排除其他岸边位置'
    result+='\n\n**V4是否仍值得进一步检查岸边云影问题：'+conclusion+'。**\n\n[并排证据](evidence.html) · [逐点数据](points.csv) · [冻结输入与模型](input_model_lock.json) · [哈希清单](SHA256SUMS.json)。\n\n点位来自AI真彩色判读，不是独立真值；计数不表示AP、准确率、正式FPR或召回。单一案例不能推及总体，原文设置未知且训练交集未知。\n'
    (out/'RESULT.md').write_text(result,encoding='utf8')
    write(out/'SHA256SUMS.json',dict(files={p.name:sha(p) for p in out.iterdir() if p.is_file()},large_arrays={k:{'shadow':v['shadow_sha256'],'classes':v['classes_sha256'],'execution_root':str(ROOT)} for k,v in state['versions'].items()}))
    print(json.dumps({'delivery':str(out),'files':len(list(out.iterdir())),'candidate_counts':{g:{v:count(g,v) for v in ['V1','V4']} for g in changed}},ensure_ascii=False))
if __name__=='__main__':main()
