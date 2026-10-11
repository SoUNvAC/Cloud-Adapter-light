"""Finish recorded exposure audit using only Catalogue/ALCD shared run logs.

Never enter old sealed/work_dirs logs, never parse/compute model metrics. New
quality-review records are not treated as model performance exposure.
"""
import csv,hashlib,json,re
from pathlib import Path
from audit_phase86 import REPO,SHARED,DATA,ALCD,ROOT,allowed,sha,canonical

def main():
    manifest=json.loads((ALCD/'imagery/product_manifest.json').read_text())
    aliases={s['key']:{s['key'],s['parameters']['cloudy_product_name'],canonical(s['gcs_product_prefix'])['canonical_product']} for s in manifest['scenes']}
    paths=set(DATA.rglob('*.log'));paths.update((ALCD/'run_logs').glob('*.log'));paths.update((ALCD/'imagery/run_logs').glob('*.log'))
    for d in SHARED.iterdir():
        m=re.match(r'phase(\d+).*control',d.name)
        if d.is_dir() and m and 65<=int(m.group(1))<=85:paths.update(d.rglob('*.log'))
    files=[];model_hits=[];quality_hits=[];errors=[]
    for p in sorted(paths):
        if any(x in str(p).lower() for x in ['sealed','target_test','backup']):continue
        try:
            raw=allowed(p).read_bytes();text=raw.decode('utf-8',errors='replace')
            matches={k:sorted(a for a in aa if a in text) for k,aa in aliases.items() if any(a in text for a in aa)}
            row=dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),matched_products=matches)
            files.append(row)
            if matches:(quality_hits if p.is_relative_to(ALCD) else model_hits).append(row)
        except OSError as e:errors.append(dict(path=str(p),error=str(e)))
    # Only already-existing manual-review submissions; no semantic interpretation.
    reviews=[]
    for p in sorted((ALCD/'imagery/visual_review_v1').glob('review_results*.csv')):
        with p.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        reviews.append(dict(path=str(p),sha256=sha(p),rows=len(rows),nonempty_judgment_fields=sum(bool(r.get('judgement','').strip()) for r in rows),is_template='template' in p.name))
    result=dict(status='completed_no_recorded_model_log_hits' if not model_hits and not errors else 'pending_log_hits_or_errors',
      candidate_lock_sha256=sha(ROOT/'candidate_lock.json'),files=files,model_log_hits=model_hits,quality_log_hits=quality_hits,errors=errors,manual_review_records=reviews,
      note='No metric values extracted or computed. Quality-only logs/panels do not prove actual human judgment; undocumented/manual/model reads remain unknown.',
      old_sealed_logs_read=False,work_dirs_logs_read=False,model_inference=False,label_pixels_read=False,code_sha256=sha(Path(__file__)))
    with (ROOT/'log_metadata_review.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,ensure_ascii=False)
    print(json.dumps(dict(status=result['status'],logs=len(files),model_log_hits=len(model_hits),quality_log_hits=len(quality_hits),errors=errors,manual_review_records=reviews)))
    if model_hits or errors:raise RuntimeError('Additional exposure evidence needs resolution before delivery')

if __name__=='__main__':main()
