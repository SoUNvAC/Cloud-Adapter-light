"""Independent small-delivery integer and tail-statistic acceptance, numpy only."""
import argparse,hashlib,json,struct,zlib
from pathlib import Path
import numpy as np


def verify(root):
    read=lambda n:json.loads((root/n).read_text(encoding='utf-8-sig'))
    digest=lambda n:hashlib.sha256((root/n).read_bytes()).hexdigest()
    manifest=read('DELIVERY_SHA256.json')
    for n,h in manifest.items():assert digest(n)==h,n
    lock=read('input_lock.json');report=read('tail_report.json');chain=read('T31TGF_chain_acceptance.json');refs=read('training_reference_lock.json');cpu=read('CPU_VERIFIED.json');strata=read('fixed_strata.json')
    for k in ['new_fits','new_thresholds','new_network_forwards','new_downloads','new_ALCD_pixel_reads']:assert report[k]==lock[k]==0
    assert chain['status']=='passed' and chain['same60_valid']==115600 and chain['reference60_counts']==[115600,0,0]
    assert chain['full20_valid']==1044484 and chain['reference20_counts']==[1044484,0,0]
    np.testing.assert_allclose(chain['three_G7_FPR_reproduced'],[.4337629757785467,.5696280276816609,.7063235294117646],rtol=0,atol=1e-14)
    assert len(chain['independent'])==2 and all(a['full_independent_scoring'] and a['all_frozen_decisions_exact'] and a['pixels']==1044484 for a in chain['independent'])
    assert len(refs['rows'])==refs['fit_groups']==100 and len(refs['features'])==7
    assert refs['rows']==[{k:a[k] for k in ['product','group_id','sample_count','sample_indices_sha256']} for a in lock['fit']]
    assert refs['sample_count']==sum(a['sample_count'] for a in refs['rows'])
    assert all(e<=1e-12 for e in refs['independent_quantile_errors'].values())
    assert read('input_score_lock.json')['indices_after']==lock['indices_before']
    assert read('input_score_lock.json')['input_lock_sha256']==digest('input_lock.json')
    assert read('input_score_lock.json')['training_reference_lock_sha256']==digest('training_reference_lock.json')
    def close(a,b):
        if b is None:assert a is None
        else:assert a is not None and abs(a-b)<1e-12,(a,b)
    rows=report['rows'];assert len(rows)==1020 and len(report['summaries'])==216
    for a in rows:
        assert a['false_positive']==a['surface_false_positive']+a['cloud_false_positive'] and a['negative']==a['surface']+a['cloud']
        close(a['fpr'],a['false_positive']/a['negative'] if a['negative'] else None)
        close(a['surface_fpr'],a['surface_false_positive']/a['surface'] if a['surface'] else None)
        close(a['cloud_fpr'],a['cloud_false_positive']/a['cloud'] if a['cloud'] else None)
        close(a['recall'],a['true_positive']/a['positive'] if a['positive'] else None)
        if a['product']==chain['product']:assert a['ap'] is None and a['recall'] is None
    for s in report['summaries']:
        selected=[a for a in rows if all(a[k]==s[k] for k in ['cohort','scale','model','threshold_name'])]
        assert len(selected)==(53 if s['cohort']=='calibration' else 32)
        if s['stratum']=='with_reference_shadow':selected=[a for a in selected if a['positive']]
        if s['stratum']=='no_reference_shadow':selected=[a for a in selected if not a['positive']]
        eligible=[a for a in selected if a[s['rate']] is not None];x=np.array([a[s['rate']] for a in eligible]);total=float(x.sum())
        assert s['groups']==len(selected) and s['defined_groups']==len(x)
        close(s['mean'],float(x.mean()) if len(x) else None);close(s['median'],float(np.median(x)) if len(x) else None)
        close(s['p90'],float(np.quantile(x,.9)) if len(x) else None);close(s['max'],float(x.max()) if len(x) else None)
        assert s['groups_above_1percent']==int((x>.01).sum())
        descending=np.sort(x)[::-1]
        close(s['top1_share_of_group_rate_sum'],float(descending[0])/total if total else None)
        close(s['top3_share_of_group_rate_sum'],float(descending[:3].sum())/total if total else None)
        close(s['mean_without_worst'],float(descending[1:].mean()) if len(x)>1 else None)
        assert s['full_mean_primary'] and s['leave_one_out_descriptive_only']
    assert len(strata['input_strata'])==32*3*7 and len(strata['source_bins'])==32*3
    mains={}
    for a in rows:
        if a['cohort']=='development' and a['scale']=='original20' and a['threshold_name']=='T20_full':mains[a['product'],a['model']]=a
    for a in strata['input_strata']:
        assert 0<=a['below_p1_count']<=a['count'] and 0<=a['above_p99_count']<=a['count']
        close(a['below_p1_fraction'],a['below_p1_count']/a['count'] if a['count'] else None)
        close(a['above_p99_fraction'],a['above_p99_count']/a['count'] if a['count'] else None)
        m=mains[a['product'],'G7_100'];expected=m['positive']+m['negative'] if a['subset']=='all_valid' else m['false_positive'] if a['subset']=='nonshadow_G7_false_positive' else m['negative']-m['false_positive']
        assert a['count']==expected
    for product in {a['product'] for a in strata['source_bins']}:
        bins=[a for a in strata['source_bins'] if a['product']==product];assert len(bins)==3
        for n in ['G7_100','SourceS_100']:
            assert sum(a['negative_count'] for a in bins)==mains[product,n]['negative']
            assert sum(a[n+'_false_positive'] for a in bins)==mains[product,n]['false_positive']
            for a in bins:close(a[n+'_fpr'],a[n+'_false_positive']/a['negative_count'] if a['negative_count'] else None)
    assert len(cpu['rows'])==170 and all(a['max_error']<=(6e-8 if a['model']=='G7_100' else 1e-4) and a['mean9_error']<=2e-15 for a in cpu['rows'])
    for n in manifest:
        if n.endswith('.png'):
            data=(root/n).read_bytes();assert data[:8]==b'\x89PNG\r\n\x1a\n';pos=8;packed=[]
            while pos<len(data):
                length=struct.unpack('>I',data[pos:pos+4])[0];tag=data[pos+4:pos+8];b=data[pos+8:pos+8+length];crc=struct.unpack('>I',data[pos+8+length:pos+12+length])[0]
                assert crc==zlib.crc32(tag+b)&0xffffffff
                if tag==b'IHDR':w,h,depth,color,_,_,_=struct.unpack('>IIBBBBB',b);assert depth==8 and color==2
                if tag==b'IDAT':packed.append(b)
                pos+=12+length
            assert len(zlib.decompress(b''.join(packed)))==h*(1+3*w)
            assert (w,h)==((340,340) if 'positive60_' in n else (1022,1022))
    return dict(verified=True,sealed_files=len(manifest),metric_rows=len(rows),tail_summaries=216,input_strata_rows=672,source_bin_rows=96,score_probe_rows=170,target_full_independent_pixels_per_model=1044484,new_fits=0,new_thresholds=0)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);print(json.dumps(verify(a.parse_args().root),indent=2))
