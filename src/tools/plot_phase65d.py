"""Readable fixed PR/FPR/CDF figures; no selection or access to raw heldout labels."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

STYLES={'source':('#0072B2','-'),'msre':('#D55E00','--')}


def setup():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':9,
                         'axes.titlesize':9,'legend.fontsize':7,'pdf.fonttype':42,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300})


def audit(fig):
    fig.canvas.draw();renderer=fig.canvas.get_renderer();width,height=fig.canvas.get_width_height()
    issues=[]
    # Matplotlib keeps Text objects for locator ticks outside the displayed limits.
    # Those ticks are not drawn; audit only the ticks that actually appear.
    hidden_ticks=set()
    for ax in fig.axes:
        for axis in [ax.xaxis,ax.yaxis]:
            lo,hi=sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                if not lo<=tick.get_loc()<=hi:
                    hidden_ticks.update([id(tick.label1),id(tick.label2)])
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text() or id(t) in hidden_ticks:continue
        box=t.get_window_extent(renderer)
        if box.x0 < -2 or box.y0 < -2 or box.x1>width+2 or box.y1>height+2:issues.append(t.get_text())
        if t.get_fontsize()<6:issues.append('small_font:'+t.get_text())
    if issues:raise ValueError('Figure layout audit: '+str(issues))


def save(fig,dest,name,final):
    audit(fig)
    if not final:
        fig.savefig(dest/(name+'_preview.png'),dpi=180)
        # Grayscale has the same line/marker encodings; retain editable originals.
        from PIL import Image
        with Image.open(dest/(name+'_preview.png')) as im:im.convert('L').save(dest/(name+'_gray.png'))
    else:
        for ext in ['png','pdf','svg']:
            path=dest/(name+'.'+ext)
            if path.exists():raise FileExistsError('Preserve final figure: '+str(path))
            fig.savefig(path,dpi=300)
    plt.close(fig)


def pr_panel(ax,data,report,cohort):
    for model in ['source','msre']:
        c=data[cohort+'_'+model+'_score'];color,style=STYLES[model]
        ax.step(c['recall'],c['precision'],where='pre',color=color,linestyle=style,lw=1.2,label=f'{model.upper()} AP={c["ap"]:.3f}')
        op=report['cohorts'][cohort][model]['original']
        ax.plot(op['recall'],op['precision'],marker='o' if model=='source' else '^',color=color,ms=5,ls='none')
    fixed=report['cohorts'][cohort]['msre']['frozen_correction']
    if fixed['precision'] is not None:ax.plot(fixed['recall'],fixed['precision'],'D',color='black',ms=5,label='Frozen correction')
    ax.set(xlim=(0,1),ylim=(0,1.02),xlabel='Shadow recall',ylabel='Shadow precision')
    ax.grid(alpha=.18);ax.legend(loc='upper right')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--final',action='store_true');a=p.parse_args()
    root=a.root;data=json.loads((root/'plot_data.json').read_text());report=json.loads((root/'diagnostic_report.json').read_text())
    out=root/'figures';out.mkdir(exist_ok=True);setup()
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.5),layout='constrained')
    for ax,cohort,title in zip(axes,['development_val','phase65b_evidence'],['a  Development (32 scenes)','b  Previously inspected 65b (64 scenes)']):
        pr_panel(ax,data,report,cohort);ax.set_title(title,loc='left')
    fig.supxlabel('All valid pixels pooled; circles/triangles: original argmax; diamond: development-frozen decision.',fontsize=7)
    save(fig,out,'pr_comparison',a.final)
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.5),layout='constrained')
    for ax,cohort,title in zip(axes,['development_val','phase65b_evidence'],['a  Development','b  Previously inspected 65b']):
        for model in ['source','msre']:
            points=data[cohort+'_'+model+'_score']['matched_fpr'];rates=np.array([float(k) for k in points]);recall=[points[k]['recall'] for k in points]
            color,style=STYLES[model];ax.plot(rates,recall,color=color,ls=style,marker='o' if model=='source' else '^',ms=4,label=model.upper())
        ax.set(xscale='log',xlim=(.0008,.12),ylim=(0,1),xlabel='Maximum allowed nonshadow FPR',ylabel='Shadow recall')
        ax.set_title(title,loc='left');ax.grid(alpha=.18);ax.legend(loc='lower right')
    fig.supxlabel('Oracle ranking diagnostic at fixed FPR budgets; evaluation-derived thresholds are not deployment choices.',fontsize=7)
    save(fig,out,'matched_fpr_recall',a.final)
    fig,axes=plt.subplots(2,2,figsize=(7.2,6.2),layout='constrained')
    for model in ['source','msre']:
        d=data['t18_'+model];c=d['ranking']['score'];color,style=STYLES[model]
        axes[0,0].step(c['recall'],c['precision'],where='pre',color=color,ls=style,lw=1.2,label=f'{model.upper()} AP={c["ap"]:.3f}')
        rates=[float(k) for k in c['matched_fpr']];recall=[v['recall'] for v in c['matched_fpr'].values()]
        axes[0,1].plot(rates,recall,color=color,ls=style,marker='o' if model=='source' else '^',ms=4,label=model.upper())
        for score,ax in [('score',axes[1,0]),('native_score',axes[1,1])]:
            for stratum in ['shadow','dark_surface']:
                dist=d['distributions'][score][stratum]
                if not dist['n']:continue
                line={'source':{'shadow':'-','dark_surface':'-.'},'msre':{'shadow':'--','dark_surface':':'}}[model][stratum]
                ax.plot(dist['quantiles'],dist['q'],color=color,ls=line,lw=1.2,label=f'{model.upper()} '+('shadow' if stratum=='shadow' else 'dark surface'))
    op=report['t18']['msre']['frozen_correction']
    if op['precision'] is not None:axes[0,0].plot(op['recall'],op['precision'],'D',color='black',ms=5,label='Frozen correction')
    axes[0,0].set(xlim=(0,1),ylim=(0,1.02),xlabel='Shadow recall',ylabel='Shadow precision',title='a  T18FYG PR (all valid pixels)')
    axes[0,1].set(xscale='log',xlim=(.0008,.12),ylim=(0,1),xlabel='Maximum allowed nonshadow FPR',ylabel='Shadow recall',title='b  T18FYG matched-FPR ranking')
    axes[1,0].set(xlabel='Softmax shadow score (not calibrated)',ylabel='Empirical cumulative fraction',title='c  Shadow and dark-surface score CDF',ylim=(0,1))
    axes[1,1].set(xlabel='Native aggregated shadow score',ylabel='Empirical cumulative fraction',title='d  Native shadow-score CDF',ylim=(0,1))
    for ax in axes.ravel():ax.grid(alpha=.18);ax.legend(loc='best')
    fig.supxlabel('Post-hoc fixed diagnostic: T18FYG selected after observed failure. Dark surface: reference Surface, TOA RGB mean <0.08.',fontsize=7)
    save(fig,out,'t18fyg_diagnostic',a.final)
    print('Figure layout audit passed; '+('final PNG/PDF/SVG exported' if a.final else 'preview and grayscale ready'))


if __name__=='__main__':main()
