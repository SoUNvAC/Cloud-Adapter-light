"""Diagnose the recorded verifier failure using gradient arithmetic only, no models."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

BASE=Path('/home/scv/shared/phase111_ocm_runtime')
sys.path.insert(0,str(BASE/'assets/vendor'))
from omnicloudmask.model_utils import create_gradient_mask


def main():
    original=json.loads((BASE/'single_scene_acceptance/RUN_STATUS.json').read_text())
    actual=create_gradient_mask(1000,300,torch.device('cuda'),torch.float32).cpu().numpy()
    numbers=np.minimum(np.minimum(np.arange(1,1001),np.arange(1000,0,-1)),300).astype(np.float32)
    axis=numbers/np.float32(300)
    independent=axis[:,None]*axis[None,:]
    reciprocal=numbers*np.float32(1/300)
    alternative=reciprocal[:,None]*reciprocal[None,:]
    gradient_error=float(np.max(np.abs(independent-actual)))
    assert gradient_error==original['gradient_max_abs']
    cpu=np.zeros((5490,5490),np.float32);author=torch.zeros((5490,5490),dtype=torch.float32)
    actual_tensor=torch.from_numpy(actual)
    for t,b,l,r in original['patches']:
        cpu[t:b,l:r]+=independent;author[t:b,l:r]+=actual_tensor
    error=float(np.max(np.abs(cpu-author.numpy())))
    result={'diagnosis':'independent CUDA/NumPy float32 gradient rounding, not an input or weight checksum error',
            'gradient_mask_max_abs':gradient_error,'reconstructed_gradient_field_max_abs':error,
            'reciprocal_variant_mask_max_abs':float(np.max(np.abs(alternative-actual))),
            'coverage_boolean_exactly_equal':bool(np.array_equal(cpu>0,author.numpy()>0)),
            'original_main_calls':original['main_calls'],'original_repeat_calls':original['repeat_calls'],
            'original_construction_graph_calls':len(original['construction_graph_calls']),
            'extra_model_calls':0,'no_labels_or_scores_accessed':True,
            'original_confidence_max_error':'not persisted; preceding <=1e-5 gate passed, exact value unavailable',
            'original_unlabelled_scores_sealed':False,
            'repair':'record gradient numeric error and require exact coverage equivalence; keep original confidence/repeat <=1e-5 criteria'}
    assert result['coverage_boolean_exactly_equal'] and 0<error<1e-5
    with (BASE/'GRADIENT_VERIFIER_DIAGNOSIS.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
