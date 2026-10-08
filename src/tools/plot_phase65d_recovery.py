"""Paired group evidence and actual decision FPR, no pixel sample-size inflation."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_phase65d import setup,save

NAMES=['source_original','msre_original','msre_calibration','msre_source','msre_source_spectral']
LABELS=['Source','MsRE original','L1: calibration','L2: +Source','L3: +spectra']
COLORS=['#777777','#000000','#0072B2','#D55E00','#009E73']
MARKERS=['x','+','o','^','D']
def short(p):return p.split('_')[5]

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--final',action='store_true');a=p.parse_args()
    root=a.root;d=json.loads((root/'recovery_report.json').read_text());curves=json.loads((root/'plot_data.json').read_text())
    assert d['status']=='completed_frozen_fit_development_recovery';out=root/'figures';out.mkdir(exist_ok=True);setup()
    supported=[r['product'] for r in d['policies'][NAMES[0]]['rows'] if r['ranking']['ap'] is not None]
    assert len(supported)==7
    fig,axes=plt.subplots(2,2,figsize=(7.2,6.3),layout='constrained')
    ax=axes[0,0]
    for i,name in enumerate(NAMES):
        rs={r['product']:r for r in d['policies'][name]['rows']}
        ax.scatter([rs[p]['ranking']['ap'] for p in supported],np.arange(7)+(i-2)*.12,s=20,color=COLORS[i],marker=MARKERS[i],label=LABELS[i])
    ax.set_yticks(range(7),[short(p) for p in supported]);ax.set(xlim=(0,1),xlabel='Shadow AP',title='a  Seven related-group representatives');ax.grid(axis='x',alpha=.18)
    ax=axes[0,1];pairs=d['comparisons']['msre_source_spectral_minus_msre_source']['pairs']
    ax.axvline(0,color='.6',lw=.7);ax.axhline(0,color='.6',lw=.7)
    for i,r in enumerate(pairs):
        xx=r['delta_ap']*100;yy=r['delta_recall']*100
        ax.scatter(xx,yy,s=20,color='#009E73',marker='D');ax.annotate(short(r['product']),(xx,yy),xytext=(3,3+i%2*5),textcoords='offset points',fontsize=6)
    ax.margins(.3);ax.set(xlabel='L3 minus L2 AP (percentage points)',ylabel='L3 minus L2 locked recall (pp)',title='b  Spectral increment, fixed fit thresholds');ax.grid(alpha=.18)
    ax=axes[1,0]
    for i,name in enumerate(NAMES):
        vals=np.array([r['fpr'] for r in d['policies'][name]['rows']])*100
        jitter=np.linspace(-.15,.15,len(vals));ax.scatter(i+jitter,vals,s=12,color=COLORS[i],marker=MARKERS[i],alpha=.8)
        ax.plot([i-.18,i+.18],[vals.mean()]*2,color='black',lw=1.2)
    ax.axhline(1,color='.5',ls=':',lw=.8,label='Fit budget 1%');ax.set_xticks(range(5),['Source','MsRE','L1','L2','L3'])
    ax.set(ylabel='Actual development FPR (%)',title='c  All 32 groups; bar = group mean');ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.18)
    ax=axes[1,1]
    for i,name in enumerate(NAMES):
        c=curves[name];ax.step(c['recall'],c['precision'],where='pre',color=COLORS[i],ls='--' if i<2 else '-',lw=1 if i<2 else 1.3,label=LABELS[i])
        m=d['policies'][name]
        if m['precision'] is not None:ax.plot(m['recall'],m['precision'],color=COLORS[i],marker=MARKERS[i],ls='none',ms=4)
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='Pooled Shadow recall',ylabel='Pooled precision',title='d  PR and original / fit-locked decisions');ax.legend(fontsize=6,loc='upper right');ax.grid(alpha=.18)
    fig.supxlabel('Fit: 153 groups, equal group weights. Development: 32 groups, 7 Shadow-supported. No development fitting or T18 access.',fontsize=6.5)
    save(fig,out,'three_level_recovery',a.final)
    print('Preview ready' if not a.final else 'Final exported',flush=True)

if __name__=='__main__':main()
