"""Build one appearance-only control from the frozen Phase442 pipeline."""
from pathlib import Path
P=Path(__file__).parent
s=(P/'extract_phase442.py').read_text(encoding='utf-8')
s=s.replace('phase442_20261103','phase443_20261103')
s=s.replace("rows=json.loads((R/'INPUTS.json').read_text());start=time.monotonic();signatures=[]", "rows=json.loads((R/'INPUTS.json').read_text());start=time.monotonic();signatures=[]\n    from appearance_phase443 import Alignment\n    alignment=Alignment(R) ")
s=s.replace("sample=pipe(info);values.clear()", "sample=pipe(info);values.clear()\n        sample['inputs']=torch.from_numpy(alignment.apply(sample['inputs'].numpy(),domain,row['index']))")
s=s.replace("save('SIGNATURES.json',signatures);", "alignment.finish();save('SIGNATURES.json',signatures);")
(P/'extract_phase443.py').write_text(s)
s=(P/'analyze_phase442.py').read_text(encoding='utf-8').replace('phase442_20261103','phase443_20261103')
a=s.index('        with np.load(OLD/');b=s.index('        for s in row',a)
s=s[:a]+s[b:]
a=s.index('    moments=scene_moments');b=s.index('    indices=',a)
s=s[:a]+'''    BASE=Path('/home/scv/shared/phase442_20261103')
    with np.load(R/'COMMON_PCA.npz') as p:center=p['center'];basis=p['basis']
    assert sha(R/'COMMON_PCA.npz')==sha(BASE/'COMMON_PCA.npz')
    explained=json.loads((BASE/'LOCK.json').read_text())['explained_variance']
    z=(x-center)@basis;del x
'''+s[b:]
a=s.index('    represented=');b=s.index('    result=[]',a)
s=s[:a]+"    boots=np.load(R/'SCENE_BOOTSTRAP.npy');directions=np.load(R/'SW_DIRECTIONS.npy')\n    result=[]\n    baseline={v['scope']:v for v in json.loads((BASE/'SUMMARY.json').read_text())}\n"+s[b+len('    result=[]\n'):]
s=s.replace("        result.append(out);", """        original=baseline[name]
        with np.load(BASE/(name+'_MOMENTS.npz')) as old:
            assert np.array_equal(idx,old['selected_indices']) and np.array_equal(m[2],old['scene_counts'])
            delta=bs-old['F_bootstrap']
        out.update(F_before=original['F'],F_delta=f-original['F'],F_reduction_percent=100*(1-f/original['F']),F_delta_lo=float(np.quantile(delta,.025)),F_delta_hi=float(np.quantile(delta,.975)),SW1_before=original['SW1_exact'],SW1_delta=sw['SW1_exact']-original['SW1_exact'])
        result.append(out);""")
a=s.index("    save('SUMMARY.json'")
s=s[:a]+'''    save('SUMMARY.json',result)
    save('LOCK.json',dict(model='Official DINOv3-S+/16',layer='block12 final LayerNorm patch tokens',PCA_sha=sha(R/'COMMON_PCA.npz'),PCA_refit=False,explained_variance_original=explained,input_sha=sha(R/'INPUTS.json'),alignment_sha=sha(R/'ALIGNMENT.npz'),weight_lock=json.loads((R/'official/LOCK.json').read_text()),groups=names,bootstrap='Identical Phase442 scene resamples, paired F differences; alignment and PCA fixed',conditional='Identical Phase442 purity>=90% indices; target shadow5 scenes161 tokens',analysis_sha=sha(__file__)))
    save('COMPLETE.json',dict(passed=True,forwards=567,network_updates=0,PCA_refit=False,seconds=time.monotonic()-start))
if __name__=='__main__':main()
'''
(P/'analyze_phase443.py').write_text(s)
s=(P/'collect_phase442.py').read_text().replace('phase442','phase443')
(P/'collect_phase443.py').write_text(s)
