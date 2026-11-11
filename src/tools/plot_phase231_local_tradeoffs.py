"""Plot frozen CPU diagnostic tables. Preview first; vectors on --export."""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/research_audits/phase231_local_tradeoffs_20261026'
SKILL=Path('C:/Users/Administrator/.codex/skills/scipilot-figure-skill/scripts')
sys.path.insert(0,str(SKILL))
from visual_qa import audit_layout


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--export',action='store_true');args=parser.parse_args()
    df=pd.read_csv(OUT/'BIAS_CURVES_PER_SCENE.csv')
    biases={r['model']:r['b'] for r in json.loads((ROOT/'outputs/phase231/phase231_20261026/TABLE1.json').read_text(encoding='utf-8'))}
    highlights={'LC80380362013144LGN00':('038036','#0072B2','-'),
                'LC81490372013362LGN00':('149037','#D55E00','--'),
                'LC81500412014244LGN00':('150041','#009E73','-.'),
                'LC80500162014247LGN00':('050016','#CC79A7',':')}
    plt.rcParams.update({'font.family':'Times New Roman','font.size':9,'axes.titlesize':10,
                         'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,
                         'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(3,2,figsize=(7.16,8.2),sharex=True,sharey='row')
    fields=[('mIoU','Change in scene mIoU (pp)'),
            ('Surface_to_Cloud_FPR','Change in Surface-to-Cloud FPR (pp)'),
            ('Shadow_recall','Change in Shadow recall (pp)')]
    handles={}
    for col,model in enumerate(['L8_21701','L8_21702']):
        subset=df[df.model==model]
        for row,(field,label) in enumerate(fields):
            ax=axes[row,col]
            for scene in sorted(subset.scene.unique(),key=lambda s:s in highlights):
                data=subset[subset.scene==scene].sort_values('b')
                base=data.loc[data.b==0,field].iloc[0]
                if pd.isna(base): continue
                if scene in highlights:
                    name,color,style=highlights[scene]
                    line,=ax.plot(data.b,data[field]-base,color=color,ls=style,lw=1.5,label=name,zorder=3)
                    handles[name]=line
                else:
                    line,=ax.plot(data.b,data[field]-base,color='0.72',lw=.65,zorder=1)
                    handles['Other scenes']=line
            ax.axhline(0,color='0.4',lw=.6,zorder=0)
            ax.axvline(0,color='0.65',lw=.6,ls=':')
            ax.axvline(biases[model],color='black',lw=.9,ls=(0,(3,3)))
            ax.set_xlim(-1,2.5);ax.set_xticks([-1,0,1,2])
            ax.grid(axis='y',color='0.92',lw=.5,zorder=0)
            ax.text(-.12,1.04,chr(97+row*2+col),transform=ax.transAxes,fontweight='bold',fontsize=11)
            if col==0:ax.set_ylabel(label)
            if row==0:ax.set_title(f'{model.replace("_", " / seed ")}   (frozen b={biases[model]:.3f})',pad=12)
            if row==2:ax.set_xlabel('Cloud score bias b (natural-log units)')
    names=['038036','149037','150041','050016','Other scenes']
    fig.legend([handles[k] for k in names],names,loc='lower center',bbox_to_anchor=(.51,.037),ncol=5,frameon=False,fontsize=8)
    fig.text(.5,.012,'Black dashed line: training-fitted bias. Curves: post-hoc validation diagnostics; no bias selected.',ha='center',fontsize=8)
    fig.subplots_adjust(left=.13,right=.985,top=.94,bottom=.12,hspace=.36,wspace=.15)
    fig.canvas.draw()
    issues=audit_layout(fig)
    qa={'automatic_layout_issues':issues,'width_inches':7.16,'height_inches':8.2,'panels':6,
        'data':'10 scenes, two seeds plotted separately. No uncertainty bands or new CI.',
        'legend':'Highlighted scene path/row IDs; other scenes grey. Colors plus line styles.',
        'bias_grid':'Fixed [-1,2.5] step .125 plus frozen b; not deployment selection.'}
    (OUT/'FIGURE_QA.json').write_text(json.dumps(qa,indent=2)+'\n',encoding='utf-8')
    assert not any(level=='FAIL' for level,msg in issues),issues
    fig.savefig(OUT/'BIAS_TRADEOFFS_preview.png',dpi=180)
    with Image.open(OUT/'BIAS_TRADEOFFS_preview.png') as im:
        im.convert('L').save(OUT/'BIAS_TRADEOFFS_grayscale_preview.png')
    if args.export:
        for ext in ['pdf','svg']:
            fig.savefig(OUT/f'BIAS_TRADEOFFS.{ext}')
        fig.savefig(OUT/'BIAS_TRADEOFFS.png',dpi=600)
    plt.close(fig)
    print(json.dumps(qa))


if __name__=='__main__':main()
