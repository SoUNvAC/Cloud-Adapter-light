"""Audited tables, hypothesis checks, retention and two scientific figures."""
from pathlib import Path
import csv,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from PIL import Image
R=Path('/home/scv/shared/phase441_20261103');NAMES=['surface','cloud','shadow'];MODELS=['source','adapted','official']
def js(p):return json.loads(p.read_text())
def save(n,x):(R/n).write_text(json.dumps(x,indent=2,allow_nan=False))
def csvwrite(n,rows):
    with (R/n).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def ci(x):return np.quantile(x,[.025,.975],axis=0).tolist()
def main():
    data={m:js(R/m/'analysis/RESULTS.json') for m in MODELS};mainrows=[];directions=[];nulls=[];boundary=[];pca=[]
    for m in MODELS:
        assert js(R/m/'analysis/COMPLETE.json')['passed']
        for row in data[m]:
            out=dict(model=m,position=row['position'],category=NAMES[row['category']],source_scenes=row['source_scenes'],target_scenes=row['target_scenes'],pca_explained=row['pca_explained'],small_source_gap=row['small_source_gap'])
            for metric in ['F','R','R_raw','source_gap','target_gap','shift','domain_accuracy']:
                out[metric]=row[metric];out[metric+'_lo'],out[metric+'_hi']=row[metric+'_CI']
            out.update(SW1=row['SW1'],SW1_lo=row['SW1_CI'][0],SW1_hi=row['SW1_CI'][1],SW1_exact=row['SW1_exact'],SW1_bin_error_bound=row['SW1_bin_error_bound'])
            for j in range(3):
                if j==row['category']:continue
                out['P_to_'+NAMES[j]]=row['P'][j]
                directions.append(dict(model=m,position=row['position'],category=NAMES[row['category']],toward=NAMES[j],P=row['P'][j],lo=row['P_CI'][0][j],hi=row['P_CI'][1][j]))
            # Fixed column schema for the three destinations, self-direction is blank.
            for j in range(3):out.setdefault('P_to_'+NAMES[j],None)
            mainrows.append(out)
            for kind in ['source_null','target_null','matched_cross']:
                n=row['null'][kind];nulls.append(dict(model=m,position=row['position'],category=NAMES[row['category']],kind=kind,scenes_per_side=row['null']['per_side_scenes'],F_median=n['median'],F_q025=n['q025_q975'][0],F_q975=n['q025_q975'][1]))
        boundary.extend(js(R/m/'analysis/BOUNDARY.json'))
        pca.extend(dict(model=m,**{k:v for k,v in js(p).items() if k!='model'}) for p in (R/m/'analysis').glob('*_PCA.json'))
    # Stabilize destination column order across categories.
    keys=list(mainrows[0]);mainrows=[{k:r[k] for k in keys} for r in mainrows]
    csvwrite('MAIN.csv',mainrows);csvwrite('DIRECTIONS.csv',directions);csvwrite('WITHIN_DOMAIN.csv',nulls)
    csvwrite('BOUNDARY.csv',[{**x,'P':json.dumps(x['P'])} for x in boundary]);save('PCA_SUMMARY.json',pca)
    cov=js(R/'COVERAGE.json');sampling=js(R/'SAMPLING.json');retention=[]
    for domain in ['A','B']:
        for cl in range(3):
            rows=[v for v in cov if v['domain']==domain and v['category']==cl];scenes=sorted({v['group'] for v in rows})
            ratios=[]
            for scene in scenes:
                ss=[v for v in rows if v['group']==scene];mass=sum(x['class_patch_mass'] for x in ss)
                if mass:ratios.append(sum(x['retained_class_mass'] for x in ss)/mass)
            rng=np.random.default_rng(44180+cl);bs=np.mean(rng.choice(ratios,(256,len(ratios)),replace=True),1)
            sam=[v for v in sampling if v['group'].startswith(domain+':') and v['category']==cl]
            retention.append(dict(domain=domain,category=NAMES[cl],raw_pixels=sum(x['raw_pixels'] for x in rows),
                pure_candidates=sum(x['pure_tokens'] for x in rows),mixed_candidates=sum(x['mixed_tokens'] for x in rows),
                selected_pure=sum(x['selected'] for x in sam if x['kind']=='pure'),selected_mixed=sum(x['selected'] for x in sam if x['kind']=='mixed'),
                pure_class_mass_retained=sum(x['retained_class_mass'] for x in rows)/sum(x['class_patch_mass'] for x in rows),
                scene_equal_class_mass_retained=float(np.mean(ratios)),scene_retention_lo=float(ci(bs)[0]),scene_retention_hi=float(ci(bs)[1]),
                scenes_with_class=len(ratios),scenes_with_pure=sum(x['selected']>0 for x in sam if x['kind']=='pure')))
    csvwrite('RETENTION.csv',retention)
    rank=[];interactions=[];adaptation=[]
    for m in MODELS:
        positions=list(dict.fromkeys(r['position'] for r in data[m]))
        lookup={(r['position'],r['category']):r for r in data[m]}
        for position in positions:
            values=[lookup[position,c]['R'] for c in range(3)];raw=[lookup[position,c]['R_raw'] for c in range(3)]
            with np.load(R/m/'analysis'/(position+'_BOOTSTRAP.npz')) as b:
                rank.append(dict(model=m,position=position,largest_R=NAMES[int(np.argmax(values))],largest_raw_R=NAMES[int(np.argmax(raw))],
                    shadow_largest_bootstrap_fraction=float((b['R'].argmax(1)==2).mean()),shadow_largest_raw_bootstrap_fraction=float((b['R_raw'].argmax(1)==2).mean())))
        if m=='official':continue
        for i in range(1,5):
            for after in ['attention','post']:
                pre=f's{i}_pre';post=f's{i}_{after}'
                with np.load(R/m/'analysis'/(pre+'_BOOTSTRAP.npz')) as a,np.load(R/m/'analysis'/(post+'_BOOTSTRAP.npz')) as b:
                    for cl in range(3):
                        x,y=lookup[pre,cl],lookup[post,cl];row=dict(model=m,stage=i,after=after,category=NAMES[cl])
                        for k in ['F','R','R_raw']:
                            row['delta_'+k]=y[k]-x[k];row['delta_'+k+'_lo'],row['delta_'+k+'_hi']=ci(b[k][:,cl]-a[k][:,cl])
                        for j in range(3):
                            row[f'delta_P_to_{NAMES[j]}']=y['P'][j]-x['P'][j]
                        resampling=js(R/m/'analysis/RESAMPLING.json');gn=resampling['group_names'];boot=np.array(resampling['bootstrap_counts'])
                        delta=[];bdd=[]
                        for domain in ['A:','B:']:
                            ids=[s for s in x['OOF_scene_accuracy'] if s.startswith(domain)];v=np.array([y['OOF_scene_accuracy'][s]-x['OOF_scene_accuracy'][s] for s in ids]);w=boot[:,[gn.index(s) for s in ids]]
                            delta.append(v.mean());bdd.append(w@v/w.sum(1))
                        row['delta_domain_accuracy']=float(np.mean(delta));row['delta_domain_accuracy_lo'],row['delta_domain_accuracy_hi']=ci(np.mean(bdd,0))
                        interactions.append(row)
    a={(r['position'],r['category']):r for r in data['source']};b={(r['position'],r['category']):r for r in data['adapted']}
    for key,x in a.items():
        y=b[key];adaptation.append(dict(position=key[0],category=NAMES[key[1]],source_R=x['R'],adapted_R=y['R'],source_R_raw=x['R_raw'],adapted_R_raw=y['R_raw'],source_domain_accuracy=x['domain_accuracy'],adapted_domain_accuracy=y['domain_accuracy'],scope='Compare dimensionless within-model diagnostics, never subtract feature centers across models'))
    csvwrite('H1_RANK.csv',rank);csvwrite('H2_INTERACTION.csv',interactions);csvwrite('H3_ADAPTATION.csv',adaptation)
    save('DATA_PROFILE.json',dict(main_rows=len(mainrows),models=MODELS,classes=NAMES,pca_explained_range=[min(x['pca_explained'] for x in mainrows),max(x['pca_explained'] for x in mainrows)],
        target_pure_scene_counts={x['category']:x['scenes_with_pure'] for x in retention if x['domain']=='B'},
        small_gap_rows=sum(x['small_source_gap'] for x in mainrows),missing_values='P self-directions intentionally blank; no missing main metrics',
        plot_choice='R and domain-readability curves in separate panels per model; dimensionless P heatmaps, no cross-model centroid subtraction'))
    plt.rcParams.update({'font.size':8,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#009E73','#0072B2','#D55E00'];markers=['o','s','^']
    fig,axes=plt.subplots(3,2,figsize=(7.16,8.6),dpi=300)
    for i,m in enumerate(MODELS):
        locs=[f's{j}_{p}' for j in range(1,5) for p in (['block'] if m=='official' else ['pre','post'])]+['final']
        look={(r['position'],r['category']):r for r in data[m]};xx=np.arange(len(locs))
        for cl in range(3):
            vals=[look[k,cl] for k in locs]
            for col,metric in [(0,'R'),(1,'domain_accuracy')]:
                ax=axes[i,col];yy=[v[metric] for v in vals];interval=np.array([v[metric+'_CI'] for v in vals]);scale=100 if col else 1
                ax.plot(xx,np.array(yy)*scale,color=colors[cl],marker=markers[cl],ms=3,label=NAMES[cl]);ax.fill_between(xx,interval[:,0]*scale,interval[:,1]*scale,color=colors[cl],alpha=.10)
            axes[i,0].plot(xx,[v['R_raw'] for v in vals],color=colors[cl],ls=':',lw=.9,alpha=.8)
        for col in [0,1]:
            ax=axes[i,col];ax.set_xticks(xx,locs,rotation=55,ha='right',fontsize=6);ax.grid(alpha=.2);ax.set_title(m);ax.set_ylabel('Relative domain shift R' if col==0 else 'Domain accuracy (%)')
        axes[i,1].axhline(50,color='gray',lw=.7,ls='--');axes[i,1].set_ylim(45,101)
    axes[0,0].legend(fontsize=7);fig.subplots_adjust(left=.10,right=.98,bottom=.08,top=.97,hspace=.55,wspace=.35)
    for ext in ['png','pdf','svg']:fig.savefig(R/f'domain_shift_curves.{ext}')
    plt.close(fig)
    relevant=[(0,1),(2,1),(2,0)];plots=[]
    for m in MODELS:
        locs=[f's{j}_{p}' for j in range(1,5) for p in (['block'] if m=='official' else ['pre','post'])]+['final'];look={(r['position'],r['category']):r for r in data[m]}
        plots.append((m,locs,np.array([[look[k,c]['P'][j] for c,j in relevant] for k in locs])))
    vmin=min(-.1,min(p[2].min() for p in plots));vmax=max(.1,max(p[2].max() for p in plots))
    fig,axes=plt.subplots(1,3,figsize=(7.16,4.8),dpi=300)
    for ax,(m,locs,values) in zip(axes,plots):
        im=ax.imshow(values,aspect='auto',cmap='RdBu_r',norm=TwoSlopeNorm(vmin=vmin,vcenter=0,vmax=vmax));ax.set_title(m)
        ax.set_xticks(range(3),['S -> C','Sh -> C','Sh -> S'],fontsize=7);ax.set_yticks(range(len(locs)),locs,fontsize=7)
        for i in range(len(locs)):
            for j in range(3):ax.text(j,i,f'{values[i,j]:.2f}',ha='center',va='center',fontsize=7,color='white' if abs(im.norm(values[i,j])-.5)>.3 else 'black')
    fig.subplots_adjust(left=.10,right=.86,bottom=.14,top=.93,wspace=.65);fig.text(.45,.035,'S: surface   C: cloud   Sh: shadow',ha='center',fontsize=8);cax=fig.add_axes([.89,.15,.015,.75]);fig.colorbar(im,cax=cax,label='Directional projection P')
    for ext in ['png','pdf','svg']:fig.savefig(R/f'direction_map.{ext}')
    plt.close(fig)
    for n in ['domain_shift_curves','direction_map']:Image.open(R/(n+'.png')).convert('L').save(R/(n+'_gray.png'),dpi=(300,300))
    lines=['# Phase441：类条件域偏移','',
        '535张CloudSEN验证图按107个ROI分组，Catalogue固定32组；三模型共1701次冻结前向。仅拟合PCA与线性域探针。主分析纯度≥90%，每场景每类最多128token。',
        'Source既有BN曾用无标签目标fit图校准；0目标标签监督不等于0目标图像接触。两域保留既有RGB预处理。','',
        '|模型/位置|类别|高斯距离F|R [95%区间]|原维R|P→地表 / 云 / 影|域准确率 [95%区间]|','|---|---|---|---|---|---|---|']
    for m in MODELS:
        for x in data[m]:
            if x['position'] not in ['s1_pre','s1_attention','s1_post','s1_block','final']:continue
            pc=' / '.join('—' if j==x['category'] else f'{v:.3f}' for j,v in enumerate(x['P']))
            lines.append(f"|{m}/{x['position']}|{NAMES[x['category']]}|{x['F']:.3f}|{x['R']:.3f} [{x['R_CI'][0]:.3f}, {x['R_CI'][1]:.3f}]|{x['R_raw']:.3f}|{pc}|{x['domain_accuracy']*100:.2f}% [{x['domain_accuracy_CI'][0]*100:.2f}, {x['domain_accuracy_CI'][1]*100:.2f}]|")
    lines+=['','完整主表MAIN.csv，类比例与筛选保留率RETENTION.csv/PROTOCOL.json，方向DIRECTIONS.csv，同域对照WITHIN_DOMAIN.csv，三假设H1/H2/H3表。',
        '图中实线为共同64维PCA的R，点线为原维R，阴影为场景重抽样区间。官方层为Transformer块3/6/9/12，无交互前后之分。CNN两输出另列主表。',
        '各模型/阶段分别拟合共同坐标；F跨模型或跨阶段绝对值不可直接比较。阶段内pre/attention/post共享PCA，可配对比较。P只是中心投影，不是真实分类边界。',
        '256次bootstrap以源ROI和目标组为单位，描述性PCA固定；探针区间对固定OOF场景准确率重抽样。每折PCA与标准化仅用训练场景，固定L2=0.01的逻辑回归，无超参搜索。',
        '切片W1使用64个固定方向；主值/区间基于512箱积分，同时报告精确加权W1及量化误差上界。所有方向、PCA和探针参数保留。',
        '同域对照100次随机不重叠分组，每侧场景数与额外matched_cross一致；不要拿全量跨域F直接与更小样本的同域区间判显著。混合patch用同一纯patch PCA另列补充几何，不混入主分析。',
        '区间未做多重比较校正，目标组数尤其云影有限；535图不能当535个独立场景。ROI绑定沿用既有验证，未重新认证地理独立性。']
    (R/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save('REPORT_COMPLETE.json',dict(passed=True,main_rows=len(mainrows),boundary_rows=len(boundary),network_updates=0))
if __name__=='__main__':main()
