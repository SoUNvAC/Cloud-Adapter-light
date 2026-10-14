"""Seal actual anonymous opinions before any controlled reveal; no relabel/evaluation."""
import argparse,json,hashlib,datetime,shutil,html
from pathlib import Path
R=Path(__file__).resolve().parents[2];O=R/'src/research_results/phase114_20261014';U=R/'outputs/phase114/phase114_20261014';D=R/'data/pixbox_s2_cmix_5036991/phase114_20261014'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
 p=argparse.ArgumentParser();p.add_argument('opinions',type=Path);a=p.parse_args();path=a.opinions.resolve();assert path.is_relative_to(R)
 opinion=read(path);lock=read(O/'SAMPLING_BLIND_LOCK.json');template=read(U/'blind_review_template.json');expected={r['anonymous_id'] for r in template['records']};records=opinion['records'];assert len(records)==len(expected) and {r['anonymous_id'] for r in records}==expected
 done=[r for r in records if r['status']=='reviewed'];assert done,'All-pending template cannot unlock reveal'
 assert all(r['status'] in ['reviewed','draft','not_reviewed'] for r in records)
 for r in done:
  assert r['point_candidate_class'] in ['Cloud','Shadow','Surface','混合','无法判断'] and r['cell_20m_state'] in ['同质','混合','无法判断']
  assert r['visible_evidence'].strip() and r['confidence'] in ['高','中','低']
  assert r['larger_context_needed'] in ['是','否','不确定'] and r['human_arbitration_needed'] in ['是','否','不确定']
 assert not (O/'OPINION_SHA_SEAL.json').exists(),'Preserve prior sealed opinion and reveal'
 dest=D/'SEALED_BLIND_OPINIONS.json';assert not dest.exists();shutil.copyfile(path,dest)
 seal={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'opinion_sha256':sha(dest),'reviewed':len(done),'not_reviewed_or_draft':len(records)-len(done),'before_private_reveal_read':True,'sampling_lock_sha256':sha(O/'SAMPLING_BLIND_LOCK.json'),'AI_opinion_is_not_human_confirmation':opinion.get('review_mode')=='ai'}
 with (O/'OPINION_SHA_SEAL.json').open('x',encoding='utf-8') as f:json.dump(seal,f,indent=2)
 assert sha(dest)==seal['opinion_sha256']
 private=D/'SAMPLE_REVEAL_PRIVATE.json';assert sha(private)==lock['private_reveal_sha256']
 mapping={r['anonymous_id']:r for r in read(private)['samples']}
 # Only already-reviewed records are revealed; remaining samples stay blind.
 revealed=[{'opinion':r,'reference_and_fixed_decision':mapping[r['anonymous_id']],'candidate_explanation_not_adjudicated':True} for r in done]
 with (D/'REVEALED_COMPARISON_PRIVATE.json').open('x',encoding='utf-8') as f:json.dump({'rows':revealed,'labels_unchanged':True,'performance_not_recomputed':True},f,ensure_ascii=False,indent=2)
 page='<meta charset="utf-8"><h1>已封存意见的揭示对照</h1><p>候选解释不等于人工裁决。参考和性能均未修改。</p>'
 for r in revealed:page+='<h2>'+html.escape(r['opinion']['anonymous_id'])+'</h2><pre>'+html.escape(json.dumps(r,ensure_ascii=False,indent=2))+'</pre>'
 (D/'REVEALED_COMPARISON_PRIVATE.html').write_text(page,encoding='utf-8')
 print('Opinions sealed; only',len(done),'reviewed records revealed in controlled local data; no metrics or relabeling')
if __name__=='__main__':main()
