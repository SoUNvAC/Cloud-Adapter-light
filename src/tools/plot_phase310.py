"""Prespecified single-run loss/gradient traces; no smoothing or uncertainty claim."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from phase310_common import *
from profile_phase310 import profile_data,render_report
from style_phase310 import setup_style
from visualqa_phase310 import audit_layout
from checkfigure_phase310 import check_figure
def plot():
    info=profile_data(str(ROOT/'TRAINING_CURVE.csv'))
    (ROOT/'CURVE_PROFILE.md').write_text(render_report(info),encoding='utf-8')
    setup_style(journal='general',lang='en',use_sciplots=False)
    plt.rcParams['savefig.bbox']=None
    rows=read(ROOT/'TRAINING_CURVE.csv');x=np.array([int(r['attempt']) for r in rows]);updated=np.array([r['updated']=='True' for r in rows])
    assert len(rows)==1000 and int(updated.sum())==955
    fig,axes=plt.subplots(3,1,figsize=(7.2,6.2),sharex=True,layout='constrained')
    for ax,keys,title in zip(axes,[('PL','total'),('SP',),('lora_grad_norm',)],['Accepted CE and total','Source-preserving KL','LoRA gradient norm']):
        for k,color,style in zip(keys,['#0072B2','#D55E00'],['-','--']):
            values=np.array([float(r[k]) if r[k]!='NA' else np.nan for r in rows]);ax.plot(x,values,label=k,color=color,ls=style,lw=.7)
        ax.set_title(title,loc='left',fontsize=9);ax.set_ylabel('loss' if ax is not axes[2] else 'L2 norm');ax.set_ylim(bottom=0);ax.legend(loc='upper left',bbox_to_anchor=(1.01,1),frameon=False);ax.grid(alpha=.2)
    axes[-1].set_xlabel('Original attempted step (45 empty-A steps are gaps)')
    fig.canvas.draw();issues=audit_layout(fig)
    assert not any(sev=='FAIL' for sev,msg in issues),issues
    png=ROOT/'TRAINING_CURVES.png';fig.savefig(png,dpi=300)
    im=Image.open(png);im.convert('L').save(ROOT/'TRAINING_CURVES_gray.png',dpi=(300,300))
    checks,meta=check_figure(str(png),min_dpi=299,target_inches=(7.2,6.2));assert not any(sev=='FAIL' for sev,msg in checks),checks
    fig.savefig(ROOT/'TRAINING_CURVES.svg');plt.close(fig)
    write(ROOT/'FIGURE_QA.json',dict(profile_rows=info['n_rows'],updated=955,skipped=45,selection='ordered raw traces; separate panels, alternative scatter not needed; no dual axes or smoothing',layout_issues=issues,file_checks=checks,metadata=meta,single_seed_no_error_bars=True,missing_values_are_45_skips=True,PNG_and_grayscale_for_visual_review=True))
