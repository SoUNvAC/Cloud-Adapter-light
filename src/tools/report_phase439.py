"""Pool paired counts, expose scene heterogeneity, and draw dose-response curves."""
from pathlib import Path
import csv
import json
import math
import statistics
import subprocess
import sys
import numpy as np

R=Path('/home/scv/shared/phase439_20261103')


def writecsv(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def rate(n,d):return 100*n/d if d else None


def transfer_stats(t):
    t=np.asarray(t,np.int64);correct=sum(t[i,i,:].sum() for i in range(3))
    harmed=sum(t[i,i,:].sum()-t[i,i,i] for i in range(3))
    repaired=sum(t[i,:,i].sum()-t[i,i,i] for i in range(3))
    changed=sum(t[:,i,j].sum() for i in range(3) for j in range(3) if i!=j)
    return dict(valid_pixels=int(t.sum()),baseline_correct=int(correct),harmed=int(harmed),repaired=int(repaired),
        damage_rate=rate(harmed,correct),repair_rate=rate(repaired,t.sum()-correct),
        net_error_increase=int(harmed-repaired),wrong_to_different_wrong=int(changed-harmed-repaired),
        new_surface_cloud=int(t[0,0,1]),new_surface_shadow=int(t[0,0,2]),new_cloud_shadow=int(t[1,1,2]),
        new_surface_cloud_rate=rate(t[0,0,1],t[0].sum()),new_surface_shadow_rate=rate(t[0,0,2],t[0].sum()),
        new_cloud_shadow_rate=rate(t[1,1,2],t[1].sum()))


def main():
    mainrows=[];scene_rows=[];feature_rows=[];transition_rows=[];feature_summary=[]
    totals={}
    for arm in ('source','adapted'):
        complete=json.loads((R/arm/'COMPLETE.json').read_text());assert complete['passed']
        totals[arm]=json.loads((R/arm/'SUMMARY.json').read_text())
        base=totals[arm]['original']
        paired=[json.loads(p.read_text()) for p in sorted((R/arm/'paired').glob('*.json'))]
        assert len(paired)==512
        originals={x['product']:x for x in paired if x['condition']['id']=='original'}
        for cid,total in totals[arm].items():
            group=[x for x in paired if x['condition']['id']==cid]
            assert len(group)==32
            row=dict(weight=arm,condition=cid,family=total['condition']['family'],value=total['condition']['value'],tier=total['condition']['tier'],
                mIoU=total['mIoU'],delta_mIoU=total['mIoU']-base['mIoU'],surface_IoU=total['IoU'][0],cloud_IoU=total['IoU'][1],shadow_IoU=total['IoU'][2],
                delta_surface=total['IoU'][0]-base['IoU'][0],delta_cloud=total['IoU'][1]-base['IoU'][1],delta_shadow=total['IoU'][2]-base['IoU'][2],**transfer_stats(total['transition']))
            denom=sum(x['appearance']['evaluation_pixels'] for x in group)
            row['newly_clipped_rate']=rate(sum(x['appearance']['newly_clipped_evaluation_pixels'] for x in group),denom)
            row['upper_saturation_rate']=0.
            row['newly_above_reflectance1_rate']=rate(sum(x['appearance']['newly_above_reflectance1_pixels'] for x in group),sum(x['appearance']['image_valid_pixels'] for x in group))
            mainrows.append(row)
        for d in paired:
            c=d['condition'];base_scene=originals[d['product']]
            support=np.asarray(d['metrics']['reference'])>0
            scene_score=float(np.mean([v for i,v in enumerate(d['metrics']['IoU']) if support[i]]))
            base_score=float(np.mean([v for i,v in enumerate(base_scene['metrics']['IoU']) if support[i]]))
            scene_rows.append(dict(weight=arm,condition=c['id'],family=c['family'],value=c['value'],tier=c['tier'],product=d['product'],
                fixed_support_mIoU=scene_score,delta_fixed_support_mIoU=scene_score-base_score,
                shadow_IoU=d['metrics']['Shadow_IoU'],**transfer_stats(d['transition']),
                newly_clipped_rate=rate(d['appearance']['newly_clipped_evaluation_pixels'],d['appearance']['evaluation_pixels'])))
            t=d['transition']
            for y in range(3):
                for p in range(3):
                    for q in range(3):
                        transition_rows.append(dict(weight=arm,condition=c['id'],product=d['product'],truth=y,original_prediction=p,changed_prediction=q,pixels=t[y][p][q]))
            for module,values in d['features'].items():
                feature_rows.append(dict(weight=arm,condition=c['id'],family=c['family'],tier=c['tier'],product=d['product'],module=module,**values))
    for arm in ('source','adapted'):
        for cid in totals[arm]:
            for stage in range(1,5):
                for suffix in ('post_attention','post_refinement'):
                    module=f'stage{stage}.{suffix}'
                    group=[x for x in feature_rows if x['weight']==arm and x['condition']==cid and x['module']==module]
                    amps=[x['relative_amplification_vs_pre'] for x in group if x.get('relative_amplification_vs_pre') is not None]
                    feature_summary.append(dict(weight=arm,condition=cid,module=module,scenes=len(group),valid_amplification_scenes=len(amps),
                        median_relative_change=statistics.median(x['relative_change'] for x in group),
                        median_cosine=statistics.median(x['cosine'] for x in group),
                        median_amplification=statistics.median(amps) if amps else None,
                        amplified_scene_fraction=sum(x>1 for x in amps)/len(amps) if amps else None))
    advantages=[]
    original_gap=totals['adapted']['original']['mIoU']-totals['source']['original']['mIoU']
    for cid,t in totals['source'].items():
        other=totals['adapted'][cid]
        advantages.append(dict(condition=cid,**{k:t['condition'][k] for k in ('family','value','tier')},
            source_mIoU=t['mIoU'],adapted_mIoU=other['mIoU'],adapted_advantage=other['mIoU']-t['mIoU'],
            advantage_change=other['mIoU']-t['mIoU']-original_gap))
    worst=sorted([x for x in scene_rows if x['condition']!='original'],key=lambda x:x['delta_fixed_support_mIoU'])[:20]
    writecsv(R/'MAIN.csv',mainrows);writecsv(R/'SCENES.csv',scene_rows);writecsv(R/'ERROR_TRANSFERS.csv',transition_rows)
    writecsv(R/'FEATURES.csv',feature_rows);writecsv(R/'FEATURE_AMPLIFICATION.csv',feature_summary)
    writecsv(R/'ADVANTAGE.csv',advantages);writecsv(R/'WORST_SCENES.csv',worst)
    (R/'SUMMARY.json').write_text(json.dumps(dict(main=mainrows,advantages=advantages,worst_scenes=worst,feature_amplification=feature_summary),indent=2))
    profile=R/'figure_profile.py'
    with (R/'DATA_PROFILE.json').open('w') as stream:
        subprocess.run([sys.executable,str(profile),str(R/'MAIN.csv'),'--group','weight','--group','family','--json'],stdout=stream,check=True)
    plot(mainrows,advantages)
    lines=['# Phase439：同图同标签外观干预','','两套固定权重及BN，32组配对，1024次实验前向＋64次钩子验收，零训练。原图端点各类IoU误差小于0.001点，完整差异见ENDPOINT_CHECK.json；模型参数与BN状态未改变。','',
           '主表：增益单位为百分点；新增地表→云为原正确地表变云/全部真实地表；损伤率以原正确像素为分母。','',
           '|权重|外观变换|mIoU|mIoU变化|云影IoU变化|新增地表→云 %|原正确像素损伤 %|新增截断 %|',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for x in mainrows:
        lines.append(f"|{x['weight']}|{x['condition']}|{x['mIoU']:.3f}|{x['delta_mIoU']:+.3f}|{x['delta_shadow']:+.3f}|{x['new_surface_cloud_rate']:.4f}|{x['damage_rate']:.4f}|{x['newly_clipped_rate']:.4f}|")
    lines+=['','## 原图优势在温和干预下是否消失','']
    for x in advantages:
        if x['tier']=='mild':lines.append(f"- {x['condition']}: adapted−source = {x['adapted_advantage']:+.3f}点；相对原图优势变化{x['advantage_change']:+.3f}点。")
    lines+=['','## 最差场景（固定GT存在类别集合）','']
    for x in worst[:6]:lines.append(f"- {x['weight']} / {x['condition']} / {x['product']}: 场景mIoU变化{x['delta_fixed_support_mIoU']:+.3f}点；新增截断{x['newly_clipped_rate']:.4f}%。")
    lines+=['','线性TOA浮点值不在255上截断；超过反射率1不是测得传感器饱和。负值归零的真实信息截断单独统计。','红蓝比是通道增益干预，不是已知白点下的物理色温。特征放大用于定位嫌疑，不作为根因证明。','图中线连接预先固定的强度，32组像素池化，不绘制未经定义的误差带；逐组离散与最差场景单独报告。']
    (R/'RESULT.md').write_text('\n'.join(lines)+'\n')
    if (R/'FAILURE.json').exists():(R/'FAILURE.json').rename(R/'REPORT_LAYOUT_FAILURE_resolved.json')
    (R/'COMPLETE.json').write_text(json.dumps(dict(passed=True,training_steps=0,bn_updates=0,forward_calls=1088,weights=2,groups=32,conditions=16)))


def plot(rows,advantages,only=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,'axes.titlesize':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    families=[('exposure',0.,'Exposure (EV)'),('contrast',1.,'Contrast multiplier'),('red_blue',1.,'Red/blue gain ratio'),('sharpen',0.,'Unsharp amount')]
    styles={'source':dict(color='#0072B2',marker='o',linestyle='-'),'adapted':dict(color='#D55E00',marker='s',linestyle='--')}
    for field,label,name in [('mIoU','mIoU (%)','absolute_curves'),('delta_mIoU','Change in mIoU (pp)','delta_curves'),('delta_shadow','Change in shadow IoU (pp)','shadow_curves'),('newly_clipped_rate','Newly clipped pixels (%)','clipping_curves')]:
        if only is not None and name!=only:continue
        fig,axes=plt.subplots(2,2,figsize=(7.16,5.2),dpi=300)
        fig.subplots_adjust(left=.14,right=.98,bottom=.12,top=.96,wspace=.50,hspace=.50)
        for ax,(family,neutral,title) in zip(axes.flat,families):
            for arm in styles:
                base=dict(next(x for x in rows if x['weight']==arm and x['condition']=='original'));base['value']=neutral
                points=sorted([x for x in rows if x['weight']==arm and x['family']==family]+[base],key=lambda x:x['value'])
                ax.plot([x['value'] for x in points],[x[field] for x in points],label=arm,linewidth=1.3,markersize=4,**styles[arm])
            ax.axvline(neutral,color='.7',linewidth=.6);ax.axhline(0,color='.7',linewidth=.6)
            ax.set_xlabel(title);ax.set_ylabel(label);ax.grid(alpha=.2)
            if field=='mIoU':ax.set_ylim(0,100)
        axes.flat[0].legend(frameon=False,fontsize=8)
        fig.canvas.draw()
        renderer=fig.canvas.get_renderer()
        for ax in axes.flat:
            for item in [ax.xaxis.label,ax.yaxis.label,ax.title]:
                box=item.get_window_extent(renderer)
                assert box.x0>=-1 and box.y0>=-1 and box.x1<=fig.bbox.width+1 and box.y1<=fig.bbox.height+1,(name,item.get_text(),box.bounds,fig.bbox.bounds)
        fig.savefig(R/(name+'.png'),dpi='figure');fig.savefig(R/(name+'.pdf'));fig.savefig(R/(name+'.svg'))
        Image.open(R/(name+'.png')).convert('L').save(R/(name+'_gray.png'))
        plt.close(fig)
    (R/'PLOT_CHECKS.json').write_text(json.dumps(dict(label_bounds_passed=True,families_separate=True,
        colors='Okabe-Ito blue/orange plus distinct markers and dashes',aggregation='pooled pixel confusion on same32 groups',visual_inspection_pending=True)))


if __name__=='__main__':main()
