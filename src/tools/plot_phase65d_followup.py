"""Fixed trajectory, group raw points and untrained spectral screening figures."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_phase65d import setup,save

COLORS=['#0072B2','#D55E00','#009E73']
def short(p):
    parts=p.split('_');return parts[5]+' '+parts[2][:8]

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--final',action='store_true');p.add_argument('--spectral-only',action='store_true');a=p.parse_args()
    root=a.root;d=json.loads((root/'development_report.json').read_text());s=json.loads((root/'spectral_screen.json').read_text())
    assert d['status']=='development_frozen_before_t18';out=root/'figures';out.mkdir(exist_ok=True);setup()
    xs=[0,2000,3500,4000];pts=d['points']
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.7),layout='constrained')
    series=[(axes[0,0],[[r['ranking']['ap'] for r in pts],[r['macro_ap'] for r in pts]],['Pooled AP','Scene-macro AP'],(0,1),'Average precision'),
            (axes[0,1],[[r['ranking']['matched_fpr']['0.01']['recall'] for r in pts],[r['ranking']['matched_fpr']['0.05']['recall'] for r in pts]],['FPR <= 1%','FPR <= 5%'],(0,1),'Shadow recall'),
            (axes[1,0],[[r['pooled_predicted_shadow_fraction'] for r in pts],[r['macro_predicted_shadow_fraction'] for r in pts]],['Pooled valid-label grid','Macro image-valid grid'],None,'Predicted Shadow fraction'),
            (axes[1,1],[[r['miou_percent'] for r in pts]],['Checkpoint pooled mIoU'],(0,100),'mIoU (%)')]
    for i,(ax,values,names,limits,ylabel) in enumerate(series):
        for j,(v,name) in enumerate(zip(values,names)):
            ax.plot(xs[1:],v[1:],color=COLORS[j],marker='o' if j==0 else '^',ls='-' if j==0 else '--',label=name,ms=4)
            ax.plot(xs[0],v[0],color=COLORS[j],marker='o' if j==0 else '^',ls='none',ms=4)
        if i==3 and d['logged_validation_miou']:
            ax.plot([r['step'] for r in d['logged_validation_miou']],[r['miou_percent'] for r in d['logged_validation_miou']],color='.5',ls=':',label='All logged validation mIoU')
        if limits:ax.set_ylim(limits)
        else:ax.set_ylim(bottom=0)
        ax.set_xticks(xs,['Source\nbaseline','2000','3500','4000']);ax.set_ylabel(ylabel);ax.grid(alpha=.18);ax.legend(fontsize=6.5)
        ax.set_title(['a  Development AP (7 supported groups)','b  Fixed-FPR ranking diagnostic','c  Shadow prediction fraction (32 groups)','d  mIoU and retained checkpoints'][i],loc='left',fontsize=8)
    fig.supxlabel('Source is a baseline, not a target step-0 checkpoint. Earlier AP checkpoints are missing; no interpolation across that gap.',fontsize=6.5)
    if not a.spectral_only:save(fig,out,'development_trajectory',a.final)
    else:plt.close(fig)
    rows=d['structure'];fig,axes=plt.subplots(1,2,figsize=(7.2,4.4),layout='constrained')
    y=np.arange(len(rows))
    for i,step in enumerate(['2000','3500','4000']):
        axes[0].scatter([r['delta_ap'][step] for r in rows],y+(i-1)*.18,color=COLORS[i],marker=['o','^','D'][i],s=22,label=step+' steps')
    axes[0].axvline(0,color='.5',lw=.7);axes[0].set_yticks(y,[short(r['product']) for r in rows]);axes[0].set(xlabel='AP difference from Source',title='a  Within-group ranking changes');axes[0].legend(fontsize=6.5);axes[0].grid(axis='x',alpha=.18)
    target=np.array([r['delta_ap']['2000'] for r in rows]);preds=d['loo_probes']
    for i,(name,label) in enumerate([('rgb_source','RGB + Source'),('rgb_source_nir_swir','RGB + Source + NIR/SWIR')]):
        axes[1].scatter(target,preds[name]['predictions'],color=COLORS[i],marker=['o','^'][i],s=24,label=label+'\nMAE='+format(preds[name]['mae'],'.3f'))
    allv=np.concatenate([target]+[np.asarray(p['predictions']) for p in preds.values()]);lo=min(allv)-.05;hi=max(allv)+.05
    axes[1].plot([lo,hi],[lo,hi],color='.5',ls=':',lw=.8);axes[1].set(xlim=(lo,hi),ylim=(lo,hi),xlabel='Observed 2000-step AP difference',ylabel='Leave-one-group-out prediction',title='b  Seven-group exploratory prediction');axes[1].legend(fontsize=6.5);axes[1].grid(alpha=.18)
    fig.supxlabel('Repeated checkpoints are not independent runs. Fold-local scaling; fixed ridge penalty 10; no GT proportions in predictors.',fontsize=6.5)
    if not a.spectral_only:save(fig,out,'development_group_structure',a.final)
    else:plt.close(fig)
    names=['source','msre2000','negative_rgb_mean','negative_b08','negative_b11','negative_b12','negative_nir_swir_mean']
    labels=['Source','MsRE','-RGB','-B08','-B11','-B12','-NIR/SWIR\nmean']
    fig,axes=plt.subplots(1,2,figsize=(7.2,4.4),layout='constrained')
    for ax,stratum,title in zip(axes,['surface_all','surface_lowest_decile'],['a  Shadow vs all reference Surface','b  Shadow vs lowest-brightness Surface decile']):
        matrix=np.array([[r['strata'][stratum][n]['roc_auc'] for n in names] for r in s['rows']],float)
        im=ax.pcolormesh(np.arange(len(names)+1)-.5,np.arange(len(s['rows'])+1)-.5,matrix,vmin=0,vmax=1,cmap='viridis',rasterized=False)
        ax.set_ylim(len(s['rows'])-.5,-.5)
        for (i,j),v in np.ndenumerate(matrix):ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=6.5,color='black' if v>.6 else 'white')
        ax.set_xticks(range(len(names)),labels,rotation=45,ha='right',fontsize=7);ax.set_yticks(range(len(s['rows'])),[short(r['product']) for r in s['rows']],fontsize=7)
        ax.set_title(title,fontsize=8,loc='left')
    colorbar=fig.colorbar(im,ax=axes,label='ROC AUC',shrink=.8)
    # Matplotlib rasterizes long colorbars by default, even with vector cells.
    if colorbar.solids is not None:colorbar.solids.set_rasterized(False)
    fig.supxlabel('Fixed untrained band-darkness signals; no fusion, tuned weights, or model recovery. Seven reference-Shadow groups only.',fontsize=6.5)
    save(fig,out,'development_spectral_screen_vector_v2',a.final)
    print('Preview ready' if not a.final else 'Final PNG/PDF/SVG exported')

if __name__=='__main__':main()
