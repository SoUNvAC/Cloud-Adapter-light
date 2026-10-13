"""Local metadata/hash lock, before any PixBox Source inference."""
import ast
from phase105_common import *

def main():
    assert not RESULT.exists(),'Do not overwrite a Phase105 attempt'
    previous=REPO/'src/research_results/phase104_20261013'
    support=read(previous/'support_acceptance.json')
    assert support['status']=='input_acceptance_complete' and support['positive_after']==379 and support['negative_after']==8188
    assert support['minimum_support_groups']==8
    windows=read(previous/'window_index.json')['rows'];assert len(windows)==len({key(w) for w in windows})==529
    for p in (REPO/'src/tools').glob('*phase105*.py'):ast.parse(p.read_text(encoding='utf-8-sig'))
    ref=REPO/'data/pixbox_s2_cmix_5036991/phase103_20261013/deduplicated_locations.json'
    assert sha(ref)==support['local_reference_sha256']
    RESULT.mkdir(parents=True);(RESULT/'.gitattributes').write_text('* -text whitespace=cr-at-eol\n',encoding='ascii')
    write(RESULT/'task_lock.json',dict(created_utc=now(),status='locked_before_forward',
        protocol_sha256=sha(PROTOCOL),code_sha256=code(),source_checkpoint_sha256=SOURCE_SHA,original153_model_sha256=MODEL_SHA,
        prior_sha256={p.name:sha(p) for p in previous.iterdir() if p.name in ['support_acceptance.json','input_operator_lock.json',
            'window_index.json','runtime_chain_acceptance.json','object_manifest.json','input_acceptance.json']},
        phase103_reference_sha256=sha(ref),phase103_mapping_sha256=sha(REPO/'src/research_results/phase103_20261013/reference_mapping.json'),
        windows=529,repeats=[key(w) for w in repeats(windows)],max_forwards=531,repeat_tolerance=1e-5,
        feature_order=['B04','B03','B02','B08','B11','B12','Source logit (G7 only)'],
        Source_logit='float64 clip1e-6..1-1e-6; log(x)-log1p(-x); castfloat32 then castfloat64, same Phase93 lineage',
        Source_probability='original seg_logits at1020 after MMSeg postprocess; softmax over3 classes; channel2 Shadow',
        primary='G7-G6 group-equal sampled-point AP; merge products within related group first',
        gate=dict(min_groups=8,min_positive=5,min_negative=20,bootstrap=10000,seed=105013,CI_lower_gt=0,CI_width_max_pp=20),
        labels_stay_local=True,no_new_fits=True,no_new_downloads=True,no_threshold_scan=True),True)
    print('Phase105 locked:529 unique windows +2 fixed repeats; zero fits')

if __name__=='__main__':main()
