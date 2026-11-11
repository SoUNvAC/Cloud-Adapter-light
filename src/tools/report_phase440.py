"""Pooled paired metrics, fixed-pixel trajectories and first-stage interventions."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path('/home/scv/shared/phase440_20261103')
def js(p):return json.loads(p.read_text())
def write(name,rows):
    with (R/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    table=[];scenes=[];margins=[];crossings=[];paired_checks=0
    for weight in ['source','adapted']:
        assert js(R/weight/'COMPLETE.json')['passed']
        summary=js(R/weight/'SUMMARY.json');files=sorted((R/weight/'paired').glob('*.json'));assert len(files)==32
        data=[js(p) for p in files]
        ratios=[v['ratio'] for v in summary['baseline']]
        for mode,values in summary.items():
            for j,v in enumerate(values):
                delta=v['mIoU']-values[0]['mIoU']
                base=summary['baseline'][j];base_delta=base['mIoU']-summary['baseline'][0]['mIoU']
                cm=np.sum([d['records'][mode]['entries'][j]['metrics']['confusion'] for d in data],axis=0)
                assert cm.tolist()==v['confusion']
                table.append(dict(weight=weight,intervention=mode,ratio=v['ratio'],mIoU=v['mIoU'],
                    surface_IoU=v['IoU'][0],cloud_IoU=v['IoU'][1],shadow_IoU=v['IoU'][2],
                    original_mIoU=values[0]['mIoU'],delta_own_original_pp=delta,
                    delta_drop_vs_baseline_pp=delta-base_delta,
                    absolute_gain_vs_baseline_pp=v['mIoU']-base['mIoU'],
                    baseline_cloud_recovered=sum(d['records'][mode]['baseline_cloud_recovered'][j] for d in data),
                    baseline_stable_harmed=sum(d['records'][mode]['baseline_stable_harmed'][j] for d in data)))
                for cohort in ['flippers','stable','all_original_correct_surface']:
                    selected=[next(x for x in d['records'][mode]['cohorts'] if x['ratio']==v['ratio'] and x['cohort']==cohort) for d in data]
                    n=sum(x['margin']['n'] for x in selected)
                    row=dict(weight=weight,intervention=mode,ratio=v['ratio'],cohort=cohort,n=n)
                    for metric in ['margin','raw_margin','surface_score','cloud_score']:
                        row[metric+'_mean']=sum(x[metric]['sum'] for x in selected)/n if n else None
                    row.update(cloud_fraction=sum(x['cloud_pixels'] for x in selected)/n if n else None,
                        negative_margin_fraction=sum(x['negative_margin'] for x in selected)/n if n else None)
                    margins.append(row)
                crossings.append(dict(weight=weight,intervention=mode,ratio=v['ratio'],
                    first_cloud=sum(d['records'][mode]['crossing']['first_cloud'][j] for d in data),
                    adjacent_enter_cloud=0 if j==0 else sum(d['records'][mode]['crossing']['adjacent_new_cloud'][j-1] for d in data),
                    adjacent_leave_cloud=0 if j==0 else sum(d['records'][mode]['crossing']['adjacent_leave_cloud'][j-1] for d in data)))
            for d in data:
                entries=d['records'][mode]['entries'];gt=np.asarray(entries[0]['metrics']['confusion']).sum(1);present=gt>0
                for j,e in enumerate(entries):
                    fixed=float(np.asarray(e['metrics']['IoU'])[present].mean())
                    original=float(np.asarray(entries[0]['metrics']['IoU'])[present].mean())
                    scenes.append(dict(weight=weight,product=d['product'],intervention=mode,ratio=e['ratio'],
                        mIoU=e['metrics']['mIoU'],fixed_GT_class_mIoU=fixed,delta_fixed_GT_class_pp=fixed-original,
                        flippers=d['cohort_counts']['flippers'],stable=d['cohort_counts']['stable']))
    for a in sorted((R/'source/paired').glob('*.json')):
        x=js(a);y=js(R/'adapted/paired'/a.name)
        assert x['product']==y['product']
        for mode in x['records']:
            for u,v in zip(x['records'][mode]['entries'],y['records'][mode]['entries']):
                assert u['input_sha']==v['input_sha'] and u['appearance']==v['appearance']
                paired_checks+=1
    write('MAIN.csv',table);write('MARGINS.csv',margins);write('CROSSINGS.csv',crossings);write('SCENES.csv',scenes)
    # Do not average per-scene quantiles as a pooled quantile; full per-scene distributions remain in paired JSON.
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(7.16,5.2),dpi=300)
    colors=['#0072B2','#D55E00','#009E73'];styles=['-','--',':'];markers=['o','s','^']
    for col,weight in enumerate(['source','adapted']):
        for k,mode in enumerate(['baseline','attention_off','clean_context']):
            rows=[x for x in table if x['weight']==weight and x['intervention']==mode and x['ratio']<=.7]
            for ax,key in [(axes[0,col],'mIoU'),(axes[1,col],'delta_own_original_pp')]:
                ax.plot([x['ratio'] for x in rows],[x[key] for x in rows],color=colors[k],linestyle=styles[k],marker=markers[k],ms=3,label=mode)
        axes[0,col].set_title(weight);axes[0,col].set_ylabel('mIoU (%)');axes[1,col].set_ylabel('Change from own original (pp)')
        for ax in axes[:,col]:ax.set_xlim(.71,.49);ax.grid(alpha=.2);ax.set_xlabel('Red / blue gain ratio')
    axes[0,0].legend(fontsize=6.5);fig.subplots_adjust(left=.11,right=.98,bottom=.11,top=.94,wspace=.40,hspace=.40)
    for ext in ['png','pdf','svg']:fig.savefig(R/f'paired_curves.{ext}')
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(7.16,5.2),dpi=300)
    for col,weight in enumerate(['source','adapted']):
        for k,cohort in enumerate(['flippers','stable']):
            rows=[x for x in margins if x['weight']==weight and x['intervention']=='baseline' and x['cohort']==cohort and x['ratio']<=.7]
            axes[0,col].plot([x['ratio'] for x in rows],[x['margin_mean'] for x in rows],color=colors[k],marker=markers[k],ms=3,label=cohort)
        axes[0,col].axhline(0,color='gray',lw=.7);axes[0,col].set_title(weight);axes[0,col].set_ylabel('Mean surface - cloud score')
        rows=[x for x in crossings if x['weight']==weight and x['intervention']=='baseline' and x['ratio']<.7]
        axes[1,col].bar([x['ratio'] for x in rows],[x['first_cloud']/1e6 for x in rows],width=.014,color=colors[0])
        axes[1,col].set_ylabel('First cloud flips (million)')
        for ax in axes[:,col]:ax.set_xlim(.71,.49);ax.set_xlabel('Red / blue gain ratio');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7);fig.subplots_adjust(left=.12,right=.98,bottom=.11,top=.94,wspace=.42,hspace=.42)
    for ext in ['png','pdf','svg']:fig.savefig(R/f'margin_crossings.{ext}')
    plt.close(fig)
    lines=['# Phase440：偏蓝翻转与第一次空间注入诊断','','零训练；两套权重、固定BN、同32组。比例从0.70到0.50每0.02一档，另测原图1.0。',
        'attention_off只关闭第一次交叉注意力残差；clean_context只替换该次读取的空间context，使用同图原图特征，其余路径保持偏蓝输入。后者为诊断参照。','',
        '|权重|干预|比例|mIoU|原图mIoU|相对自身原图变化|落差改善|相对常规绝对收益|',
        '|---|---|---|---|---|---|---|---|']
    for x in table:
        if x['ratio'] in [1.,.7,.6,.5]:
            lines.append(f"|{x['weight']}|{x['intervention']}|{x['ratio']}|{x['mIoU']:.4f}|{x['original_mIoU']:.4f}|{x['delta_own_original_pp']:+.4f}|{x['delta_drop_vs_baseline_pp']:+.4f}|{x['absolute_gain_vs_baseline_pp']:+.4f}|")
    lines+=['','落差改善=干预的（偏蓝−原图）−常规的（偏蓝−原图）；正值表示退化减少，必须结合原图及偏蓝绝对性能判断。',
        '得分为Mask2Former非负语义得分按三类和归一化，不是额外softmax或经过校准的概率。负地表−云差不必然预测云，还需超过阴影。',
        'flippers：常规原图正确地表且0.5变云；stable：所有常规档位一直正确地表。固定像素、结果定义分组，仅作描述。逐图分位数及确定性像素轨迹在paired JSON，主表均值按像素加权。',
        '32组为已暴露development诊断集；不把像素数当独立样本量。首次翻转表包含相邻回转数量；场景表用固定GT存在类防止空类口径变化。']
    (R/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (R/'COMPLETE.json').write_text(json.dumps(dict(passed=True,forward_calls=2304,paired_inputs=paired_checks,training_steps=0,bn_updates=0)))
if __name__=='__main__':main()
