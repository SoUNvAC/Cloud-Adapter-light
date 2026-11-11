"""Build offline human review UI from the verified80-image package."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/l8_no10_visual_review_20261028'
PACKAGE = OUT / 'images_package'


def main():
    scenes = json.loads((PACKAGE / 'SCENES.json').read_text(encoding='utf-8'))
    data = json.dumps(scenes, ensure_ascii=False).replace('</', '<\\/')
    html = r'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>L8 no十景人工复核</title>
<style>body{margin:0;background:#15191e;color:#e5eaf0;font:16px/1.65 system-ui}header{padding:18px 25px;background:#25303d;position:sticky;top:0;z-index:2}main{max-width:1500px;margin:auto;padding:24px}button,select,input,textarea{font:inherit;padding:8px;background:#fff;color:#182433;border:1px solid #8091a5;border-radius:5px}button{cursor:pointer;background:#aad5fc}textarea{width:96%;min-height:75px}label{display:block;margin:10px 0}img{max-width:100%;height:auto;background:#000}a{color:#9ed8ff}.card{padding:18px;margin:20px 0;background:#202832;border-radius:9px}.note{color:#ffd992}.fields{display:grid;grid-template-columns:1fr 1fr;gap:15px}.full{grid-column:1/-1}small{color:#b9c4d1}.progress{color:#9eecbb}h2{overflow-wrap:anywhere}</style>
<header><strong>L8 no十景人工复核</strong>　<select id="sceneSelect"></select> <button onclick="go(-1)">上一景</button> <button onclick="go(1)">下一景</button> <button onclick="download()">导出本次判读JSON</button><span class="progress" id="progress"></span></header>
<main><p>共10景、80张图。每景8张：整景RGB＋整景参考图＋6个原尺寸局部。<b>点图可打开原尺寸并继续放大</b>。前3个局部为空间覆盖点，后3个优先选择云边暗区，均未使用模型预测；不是自动识别出的真云影。</p>
<p class="note">先看整景RGB，再看局部，最后对照参考。蓝色＝参考云影（这十景预期无蓝色）；绿色＝Clear，黄色＝Thin Cloud，白色＝Cloud，深灰＝Fill。暗区也可能是水、山影、暗地物；不能仅凭“暗”判漏标。提亮图只改变显示亮度。</p>
<p>“真无影”需要你检查整景，不是只看6个局部。证据不足就选“无法判断”；这份人工判读不会自动修改旧标签或历史实验结果。</p>
<div id="content"></div><div class="card"><h2>本景判读</h2><div class="fields">
<label>判读结果<select id="decision" onchange="save()"><option value="">尚未填写</option><option>真无影（人工判断）</option><option>有明确漏标云影</option><option>疑似漏标云影</option><option>无法判断</option></select></label>
<label>把握程度<select id="confidence" onchange="save()"><option value="">未填</option><option>高</option><option>中</option><option>低</option></select></label>
<label class="full"><input type="checkbox" id="whole" onchange="save()">我已放大检查整景，不仅查看6个局部。</label>
<label class="full">证据位置：例如 W4中心附近；或整景行row=3000、列col=4200。原生行列从0开始。<input id="location" oninput="save()" style="width:95%"></label>
<label class="full">可见依据／无法判断的原因<textarea id="reason" oninput="save()" placeholder="例如：云的某侧存在对应暗带、延伸方向一致；或只有大片暗水面，无法确认是云影。"></textarea></label>
<label>主审姓名或编号<input id="reviewer" oninput="save()"></label><label>遥感判读经验（可选）<input id="experience" oninput="save()"></label>
</div><p class="note" id="saved"></p></div></main>
<script>const SCENES=__SCENES__;const KEY='L8_NO10_REVIEW_20261028_v1';let index=0;let state={};try{state=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}const ids=['decision','confidence','whole','location','reason','reviewer','experience'];const sel=document.getElementById('sceneSelect');SCENES.forEach((s,i)=>sel.add(new Option((i+1)+' / '+s.scene,i)));sel.onchange=()=>{index=Number(sel.value);render()};
function esc(x){return String(x).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('"','&quot;')}
function photo(url,title){return '<div class="card"><h3>'+esc(title)+'</h3><a target="_blank" href="'+url+'"><img loading="lazy" src="'+url+'" alt="'+esc(title)+'"></a><p><a target="_blank" href="'+url+'">打开原尺寸图（可继续放大）</a></p></div>'}
function render(){let s=SCENES[index];sel.value=index;let base='images_package/'+s.folder+'/';let h='<h2>'+s.scene+'</h2><p>整景 '+s.width+' × '+s.height+'；RGB波段B4/B3/B2。原发布TIF为8位显示数据，未额外下载高位深原始影像。</p>';h+=photo(base+'01_full_scene_RGB_native.jpg','01 整景真彩色：红框W1—W6对应局部位置');h+=photo(base+'02_full_reference_native.png','02 整景参考标签（不是模型预测）');s.windows.forEach((w,k)=>{h+=photo(base+String(k+3).padStart(2,'0')+'_W'+(k+1)+'_detail.jpg','W'+(k+1)+'：row='+w.row+' col='+w.col+'，1024×1024原像素；左RGB／中提亮／右参考');h+='<small>选择方式：'+(w.selection==='spatial_coverage'?'空间覆盖':w.selection.startsWith('coverage_fallback')?'没有候选时的覆盖补位':'云边暗区候选，仅辅助定位')+'。是否处于原评价裁块：只按取样中心记录，不代表整个窗口都被评价。</small>'});document.getElementById('content').innerHTML=h;let r=state[s.scene]||{};ids.forEach(id=>{let e=document.getElementById(id);if(id==='whole')e.checked=!!r[id];else e.value=r[id]||''});document.getElementById('saved').textContent='填写后会尝试保存在此浏览器；交付前务必点击顶部导出JSON。';progress()}
function progress(){document.getElementById('progress').textContent='　已填写 '+SCENES.filter(s=>(state[s.scene]||{}).decision).length+'/10'}
function save(){let r={};ids.forEach(id=>{let e=document.getElementById(id);r[id]=id==='whole'?e.checked:e.value});r.updated_at=new Date().toISOString();state[SCENES[index].scene]=r;try{localStorage.setItem(KEY,JSON.stringify(state));document.getElementById('saved').textContent='已保存在当前浏览器。最终请导出JSON。'}catch(e){document.getElementById('saved').textContent='浏览器禁止本地保存，请及时导出JSON，避免丢失。'}progress()}
function go(d){index=Math.max(0,Math.min(9,index+d));render();window.scrollTo(0,0)}
function download(){save();let out={audit:'L8_no10_20261028',exported_at:new Date().toISOString(),human_submission:true,rows:SCENES.map(s=>({scene:s.scene,decision:'',...state[s.scene]}))};let url=URL.createObjectURL(new Blob([JSON.stringify(out,null,2)],{type:'application/json'}));let a=document.createElement('a');a.href=url;a.download='l8_no10_review_submission.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}render();</script></html>'''
    (OUT / 'START_REVIEW.html').write_text(html.replace('__SCENES__', data), encoding='utf-8')
    with (OUT / 'REVIEW_FORM.csv').open('w', encoding='utf-8-sig', newline='') as f:
        fields = ['scene', 'decision', 'confidence', 'whole_scene_checked', 'evidence_location', 'visible_evidence', 'reviewer']
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(dict(scene=s['scene']) for s in scenes)
    (OUT / 'README.md').write_text('''# L8 no十景目视审查包

打开 START_REVIEW.html，逐景查看并填写，最后导出 l8_no10_review_submission.json 即可。也可填写 REVIEW_FORM.csv，二者选一种交付。

共80张图，每景8张；整景保持TIF原尺寸，局部1024原像素。点击图片打开原尺寸，放大到100%观察。整景红框W1—W6对应6个局部。局部三栏是同位置RGB、提亮RGB、原参考标签，不含模型预测。

先检查整景RGB，再看6个局部，对照参考标签。暗水面、山影和暗地物不等于云影；要观察与云位置、投影方向和形状的对应。无法确定就保留无法判断。6个局部不是整景穷尽检查，不能单靠它们证明全景无云影。

每景填写一个结论、把握、证据位置和可见依据。若判漏标，请写W编号或原生row/col；若判真无影，请确认已放大检查整景。所有字段初始留空，不由AI代判。浏览器仅本地保存，请务必导出文件。

注意：RGB源是现有11波段8位发布TIF的B4/B3/B2，不是新下载的高位深L1原始DN。提亮只改变显示，不能恢复源数据中不存在的细节；判读包不改变旧参考标签或旧实验指标。整景可能包括原评价裁块之外的区域，局部不能与历史逐块指标直接等同。
''', encoding='utf-8')
    print('Offline page and blank CSV created; no human decisions filled.')


if __name__ == '__main__':
    main()
