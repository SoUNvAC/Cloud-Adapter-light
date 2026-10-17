"""Review existing Phase200 delivery; mark post-hoc regions without reselection/inference."""
import base64,csv,hashlib,json,re,shutil
from pathlib import Path
from PIL import Image,ImageDraw
REPO=Path(__file__).resolve().parents[2]
ROOT=REPO/'outputs/phase200/phase200_20261017';D=ROOT/'delivery'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj):p.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def main():
    assert not (ROOT/'SPATIAL_REVIEW.json').exists(),'Do not overwrite a completed review'
    original=json.loads((D/'SHA256SUMS.json').read_text(encoding='utf8'))
    for n,h in original['files'].items():assert sha(D/n)==h,n
    for name in ['points.csv','input_model_lock.json','RESOURCE_RECEIPT.json','CANDIDATE_COUNTS.json']:shutil.copyfile(D/name,ROOT/name)
    regions=[dict(id='O1',crop_pixels=[318,35,345,130],utm11N_bbox=[573860,4938900,574400,4940800],
        observation='V4 adds a north-eastern shore-adjacent Shadow strip compared with V1. RGB shows a dark coastal/terrestrial margin; real cloud shadow or terrain/water confusion cannot be distinguished reliably with this limited context.',
        status='post-hoc qualitative lead only; NOT a new reviewed point or false-positive label'),
        dict(id='O2',crop_pixels=[220,575,335,730],utm11N_bbox=[571900,4926900,574200,4930000],
        observation='V4 expands the southern water Shadow area. This may recover real shadow or add commission; existing RGB and candidate labels do not adjudicate it.',
        status='post-hoc qualitative difference; no claimed correction or error rate')]
    review=dict(regions=regions,presealed_points_unchanged=True,no_new_point_labels=True,no_new_model_calls=True,
        common_disagreement='H6 was an AI clear-shore candidate, but both versions classify it as ThickCloud. This is candidate/model disagreement, not an adjudicated error.',
        fixed_points='All 24 default classes agree between V1 and V4. Clear-only counts unchanged; 19 clear,4 mixed,1 uncertain.',
        score_change='S3 shadow confidence 0.9488904476165771 -> 0.8081921339035034, still Shadow; not a calibrated correctness probability.',
        interpretation='Evidence of a local shore-adjacent V4 difference worth further inspection; no independent truth, no exact historical reproduction, no global resolution claim.',
        original_delivery_sha256=sha(D/'SHA256SUMS.json'))
    write(ROOT/'SPATIAL_REVIEW.json',review)
    names={'true_color.png':'evidence_rgb.png','V1.png':'evidence_V1.png','V4.png':'evidence_V4.png'}
    for old,new in names.items():
        im=Image.open(D/old).convert('RGB');draw=ImageDraw.Draw(im)
        for reg in regions:
            x0,y0,x1,y1=reg['crop_pixels'];draw.rectangle((x0,y0,x1,y1),outline='#ff00ff',width=2);draw.text((x0+2,y0+2),reg['id'],fill='white',stroke_width=1,stroke_fill='black')
        im.save(ROOT/new)
    page=(D/'evidence.html').read_text(encoding='utf8')
    embeds=re.findall('data:image/png;base64,([^\"]+)',page);assert len(embeds)==3
    for payload,name in zip(embeds,names.values()):page=page.replace(payload,base64.b64encode((ROOT/name).read_bytes()).decode(),1)
    note='<p><b>Post-hoc spatial observations (magenta boxes; NOT additional sampled labels):</b> O1: V4 adds a north-eastern coastal Shadow strip, requiring context to distinguish real cloud shadow from surface/terrain confusion. O2: southern water Shadow expands; improvement versus commission is unadjudicated. H6: both versions call an AI clear-shore candidate ThickCloud. All 24 frozen point classes agree across versions; no point was reselected.</p>'
    page=page.replace('<h2>Fixed candidate points',note+'<h2>Fixed candidate points');(ROOT/'evidence.html').write_text(page,encoding='utf8')
    text=(D/'RESULT.md').read_text(encoding='utf8');lines=text.splitlines()
    for i,line in enumerate(lines):
        if line.startswith('| 原文岸边疑似误报现象 |'):
            lines[i]='| 原文岸边疑似误报现象 | 明确岸边候选0/3判Shadow；H6共同Cloud分歧 | O1东北岸新增贴岸Shadow，有待复核 | [O1并排图](evidence.html)；事后空间线索，不能确诊误报 |'
        if line.startswith('| V4相对V1新增或减少的问题 |'):
            lines[i]='| V4相对V1新增或减少的问题 | 24点类别与V4一致；H6判ThickCloud | O1东北岸新增Shadow；O2南部水面影区扩张；S3分数下降但仍判Shadow | [事后空间观察](SPATIAL_REVIEW.json)；不新增参考点，不宣称修复或新增误报已获裁决 |'
        if line.startswith('**V4是否仍值得进一步检查岸边云影问题：'):
            lines[i]='**V4是否仍值得进一步检查岸边云影问题：有线索。**'
    text='\n'.join(lines)+'\n'
    text=text.replace('点位来自AI真彩色判读，不是独立真值；计数不表示AP、准确率、正式FPR或召回。单一案例不能推及总体，原文设置未知且训练交集未知。',
        '数据结果：24点默认类别一致，明确候选计数见主表；O1/O2是事后空间差异，不是新增真值点。结论探讨：O1值得继续核查，但固定岸边点仅3个明确，不能用0/3排除其他湖岸问题，H6共同Cloud分歧也保留。AI候选不构成独立真值，不计算AP、准确率、正式FPR或召回，原文设置和训练交集未知。')
    (ROOT/'RESULT.md').write_text(text,encoding='utf8')
    with (ROOT/'points.csv').open(encoding='utf-8-sig',newline='') as f:points=list(csv.DictReader(f))
    with (ROOT/'candidate_points.csv').open(encoding='utf-8-sig',newline='') as f:prior=list(csv.DictReader(f))
    assert len(points)==len(prior)==24
    for a,b in zip(prior,points):assert all(a[k]==b[k] for k in a)
    validation=dict(status='passed',original_delivery_sha_checked=9,point_identities_checked=24,default_classes_identical=24,
        clear_mixed_uncertain=[19,4,1],visual_review='RGB,V1,V4 inspected; O1/O2 qualitative regions flagged; H6 common disagreement retained',
        changed_model_calls=0,changed_thresholds=0,changed_candidates=0,independent_ground_truth=False,formal_error_metrics_computed=False)
    write(ROOT/'DELIVERY_VALIDATION.json',validation)
    required=['RESULT.md','evidence.html','points.csv','input_model_lock.json','RESOURCE_RECEIPT.json','CANDIDATE_COUNTS.json','SPATIAL_REVIEW.json','DELIVERY_VALIDATION.json']+list(names.values())
    write(ROOT/'SHA256SUMS.json',dict(files={n:sha(ROOT/n) for n in required},large_arrays=original['large_arrays'],original_delivery_manifest_sha256=sha(D/'SHA256SUMS.json')))
    print(json.dumps(validation))
if __name__=='__main__':main()
