"""Independent standard-library arithmetic audit and concise diagnosis delivery."""
from pathlib import Path
import csv,json,hashlib
ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'outputs/phase419/phase419_20261101'
def js(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(n):return list(csv.DictReader((R/n).open(encoding='utf-8')))
def iou(c):
    return [100*c[i][i]/(sum(c[i])+sum(r[i] for r in c)-c[i][i]) if sum(c[i])+sum(r[i] for r in c)-c[i][i] else None for i in range(3)]
def mi(c):
    v=[x for x in iou(c) if x is not None];return sum(v)/len(v)
def sub(a,b):return [[a[i][j]-b[i][j] for j in range(3)] for i in range(3)]
def add(a,b):return [[a[i][j]+b[i][j] for j in range(3)] for i in range(3)]
def write(n,data):
    with (R/n).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,list(data[0]));w.writeheader();w.writerows(data)
def main():
    for n,h in js(R/'TRANSFER_MANIFEST.json').items():assert sha(R/n)==h,n
    a=js(R/'ACCEPTANCE.json');assert a['passed'] and a['new_model_forwards']==a['new_training_updates']==0
    mods={x['model']:x for x in rows('MODELS.csv')};scenes=rows('SCENE_IMPACT.csv');classes=rows('CLASS_DIAGNOSIS.csv')
    bc=json.loads(mods['Phase416']['confusion']);base=mi(bc)
    for model,m in mods.items():
        groups=[x for x in scenes if x['model']==model];assert len(groups)==32
        cm=[[sum(json.loads(g['confusion'])[i][j] for g in groups) for j in range(3)] for i in range(3)]
        assert cm==json.loads(m['confusion']) and abs(mi(cm)-float(m['mIoU']))<1e-9
        assert abs(sum(float(x['contribution_to_delta_vs_Phase416_pp']) for x in classes if x['model']==model)-(mi(cm)-base))<1e-9
        for g in groups:
            gc=json.loads(g['confusion']);bg=next(x for x in scenes if x['model']=='Phase416' and x['product']==g['product'])
            assert abs(mi(add(sub(cm,gc),json.loads(bg['confusion'])))-mi(cm)-float(g['pooled_gain_if_scene_replaced_by_Phase416_pp']))<1e-9
    for x in rows('ORACLE_SENSITIVITY.csv'):
        cm=json.loads(mods[x['model']]['confusion']);i=['surface','cloud','shadow'].index(x['repair_truth']);j=['surface','cloud','shadow'].index(x['repair_prediction']);cm[i][i]+=cm[i][j];cm[i][j]=0
        assert abs(mi(cm)-float(x['oracle_mIoU']))<1e-9
    detailed=rows('SCORE_STRATA.csv');summary=rows('SCORE_SUMMARY.csv')
    for x in summary:
        rr=[v for v in detailed if v['model']==x['model'] and v['stratum']==x['stratum']];n=sum(int(v['pixels']) for v in rr);assert n==int(x['pixels'])
        for key in ['mean_shadow_score','fraction_score_ge_0_5_pct']:
            assert abs(sum(int(v['pixels'])*float(v[key]) for v in rr)/n-float(x[key]))<1e-9
    product='S2A_MSIL1C_20180909T151911_N0206_R068_T25XEG_20180909T205122'
    sens=[]
    bgroup=next(x for x in scenes if x['model']=='Phase416' and x['product']==product);bgroupcm=json.loads(bgroup['confusion']);base31=mi(sub(bc,bgroupcm))
    for model,m in mods.items():
        g=next(x for x in scenes if x['model']==model and x['product']==product);cm=json.loads(m['confusion']);gc=json.loads(g['confusion']);rest=mi(sub(cm,gc))
        sens.append({'model':model,'official32_mIoU':mi(cm),'T25XEG_surface_to_cloud':gc[0][1],'counterfactual_replace_one_scene_with_baseline_mIoU':mi(add(sub(cm,gc),bgroupcm)),'same31_excluding_T25XEG_mIoU':rest,'same31_delta_vs_baseline':rest-base31,'note':'post-hoc sensitivity only; all32 official results retained; never select/delete test cases using this'})
    write('SINGLE_SCENE_SENSITIVITY.csv',sens)
    prim=['dual_operator','link_cloud','field_illumination']
    chinese={'dual_operator':'双算子','link_cloud':'云语义关联','field_illumination':'照明分组'}
    lines=['# Phase419：零训练诊断结果','','**核心发现：阴影识别略有改善，但伴随云—地表判断受损；总体掉点高度集中于一个高亮、低纹理场景。**','','1. 基线真正的瓶颈','','Phase416：地表IoU 95.2923%、云IoU 95.7676%、影IoU 41.1275%。阴影仅占有效像素1.9626%，但三类mIoU等权，影类贡献了距100%差距的约86.82%。这只是指标分解，不是可实现提升预测。','','基线漏掉282446个阴影像素：211902判成地表，70544判成云。误报阴影252226个，其中200265来自真值云（约79.40%）。因此同时存在阴影漏检和暗云误报，不能只降低阴影阈值。','','2. 新结构到底损失了什么','','|机制|总mIoU变化/百分点|地表贡献|云贡献|影贡献|','|---|---:|---:|---:|---:|']
    for model in prim:
        d={x['class']:float(x['contribution_to_delta_vs_Phase416_pp']) for x in classes if x['model']==model}
        lines.append(f"|{chinese[model]}|{float(mods[model]['mIoU'])-base:+.4f}|{d['surface']:+.4f}|{d['cloud']:+.4f}|{d['shadow']:+.4f}|")
    lines+=['','上表贡献为各类IoU差/3，可精确相加；不是机制因果分解。','','3. 一个场景强烈影响排名','','共同失败场景：`'+product+'`。真值1044484像素全部为地表，预存亮度均值0.7928、标准差0.00954；RGB目视高亮且低纹理。RGB本身不足以确认具体材质或判定标签对错，未改标签。','','|模型|该场景地表→云像素|官方32组mIoU|只换回该场景基线结果的反事实mIoU|','|---|---:|---:|---:|']
    for model in ['Phase416']+prim:
        s=next(x for x in sens if x['model']==model)
        lines.append(f"|{chinese.get(model,'无结构对照')}|{s['T25XEG_surface_to_cloud']}|{s['official32_mIoU']:.4f}|{s['counterfactual_replace_one_scene_with_baseline_mIoU']:.4f}|")
    lines+=['','反事实仅用于定位损失，不是可部署门控，也不能据此删掉该场景或把反事实数字作为正式成绩。SINGLE_SCENE_SENSITIVITY.csv还列出所有模型在相同剩余31组上的敏感性；官方分数始终是完整32组。机制对照仍须保留，不能把局部改进直接解释为物理机制有效。','','4. 云影的具体难点','','- T40XDR（2018-04-30）目视为雪冰地貌：基线将85502个真值阴影全部判为地表。双算子/云语义关联/照明分组分别找回4376/16457/19877个阴影，但该场景地表误报云也从1223增至30073/140123/61944。','- 阴影边界3像素范围占阴影约46.61%；其中得分≥0.5的比例36.36%，内部为71.79%。这是得分诊断，不等于三分类召回，也不能据此声称所有漏检都在边界。','- 每景真值云最暗约10%像素中，阴影得分≥0.5的比例7.25%，其余云为0.312%。应关注暗云与阴影的区分，而非只把暗像素当阴影。','- 153个训练组有40组含阴影，训练阴影像素占2.0307%；32个开发组中25组是真值纯地表或纯云，只有7组三类混合，阴影占1.9626%。全局云/地表高IoU受大量纯类像素影响；不能把不同类别集合的子集mIoU直接横比。','','5. 下一步建议（未执行）','','先核对T25XEG原始标签和原始影像；不因模型失败而擅自修改标签。之后优先试一个小而明确的结构约束：冻结当前目标基线，只让新分支修正阴影概率，并保持非阴影部分的云/地表概率比不变。这样在数学上阻止当前最严重的“原本地表变成云”副作用，再针对雪冰相对阴影、暗云拒识和细边界设计输入特征。此约束本身不是已证明的新颖性或收益。','','例：q_shadow=sigmoid(logit(p_shadow)+delta)，q_surface=(1-q_shadow)*p_surface/(p_surface+p_cloud)，q_cloud同理。实际实现需数值保护；冻结基线才能保证比例不变。阴影增益仍可能引入阴影误报，因此须沿用完整32组及匹配对照验证。','','6. 验收与限制','','11个模型、32组、352份已有得分文件。零训练、零新增前向；远端使用numpy/scipy/PIL，未加载模型。逐组混淆矩阵、真值支持、文件哈希、类别贡献和场景替换计算已复核；本地另用标准库独立重算。','','没有保存完整三类硬预测图：不能逐像素定位三分类修复/新增错误。GT分层、完美修复和场景替换属于事后诊断，不是可用标签推理。所有结论来自已暴露development、单seed，未做独立确认。RGB预览使用统一clip(RGB/255,0,1)^(1/2.2)，不是材质或标签鉴定。']
    (R/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (R/'README.md').write_bytes((R/'RESULT.md').read_bytes())
    receipt={'passed':True,'independent_standard_library_CM_check':True,'all_scene_replacement_calculations_checked':True,'oracle_CM_calculations_checked':True,'score_aggregation_checked':True,'new_model_forwards':0,'new_training_updates':0,'models':11,'groups':32}
    (R/'LOCAL_ACCEPTANCE.json').write_text(json.dumps(receipt,indent=2))
    log=ROOT/'src/EXPERIMENT_LOG.md';raw=log.read_bytes();marker='## 后续维护规则'.encode();assert raw.count(marker)==1
    title='## Phase419：零训练错误诊断（2026-11-01）'
    entry=title+'\n\n- 11模型、固定32组、352份已有得分；零训练、零新增前向，CPU诊断及本地独立数值验收通过。\n- 阴影IoU是基线瓶颈；三个结构的影类均改善，但云/地表损失抵消收益。共同掉点高度集中于T25XEG高亮低纹理、全地表真值场景。\n- T40XDR雪冰场景基线漏掉全部85502个阴影；新结构部分找回，但伴随地表误报云。\n- 场景替换、真值分层仅诊断；不删场景、不改标签，不把反事实作为成绩。建议先核查标签来源，再试保持非阴影云/地表比值的阴影专用分支；未执行。\n- 结果：outputs/phase419/phase419_20261101/RESULT.md。\n\n'
    if title.encode() not in raw:log.write_bytes(raw.replace(marker,entry.replace('\n','\r\n').encode()+marker))
    plan=ROOT/'src/research_plans/phase419.md';s=plan.read_text(encoding='utf-8');status='完成：11模型、32组、352份已有得分，零训练/零新增前向；结果与验收已存outputs/phase419/phase419_20261101。'
    if status not in s:plan.write_text(s+'\n'+status+'\n',encoding='utf-8')
    (R/'DELIVERY_COMPLETE.json').write_text(json.dumps(receipt,indent=2))
    manifest={p.relative_to(R).as_posix():sha(p) for p in R.rglob('*') if p.is_file() and p.name!='LOCAL_MANIFEST.json'}
    (R/'LOCAL_MANIFEST.json').write_text(json.dumps(manifest,indent=2));print('sealed',len(manifest),'files')
if __name__=='__main__':main()
