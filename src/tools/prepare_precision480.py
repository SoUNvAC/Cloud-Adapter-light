from pathlib import Path
T=Path(__file__).parent;s=(T/'ablate_shadow480.py').read_text(encoding='utf-8')
s=s.replace('def main():', '''def strict_projection(m,args,out):
    with torch.backends.cudnn.flags(allow_tf32=False):return torch.nn.functional.conv2d(args[0],m.weight)

def main():''')
needle='setattr(network.branch.redistribution,"enabled",False)';assert needle in s
s=s.replace(needle,'network.branch.redistribution.inject.register_forward_hook(strict_projection)')
s=s.replace("'ablation_off'","'precision_strict'").replace('OFF_BOOTSTRAP','STRICT_BOOTSTRAP').replace('OFF_INDEPENDENT','STRICT_INDEPENDENT').replace('ABLATION_OFF','STRICT_PRECISION')
s=s.replace('cm=np.zeros((3,3),np.int64);oldcm=', 'maximum_mean_residual=0.;cm=np.zeros((3,3),np.int64);oldcm=')
s=s.replace('            valid=truth!=255;', '            maximum_mean_residual=max(maximum_mean_residual,network.branch.redistribution.mean_residual.item());assert maximum_mean_residual<1e-5\n            valid=truth!=255;')
s=s.replace('state_unchanged=True)', 'state_unchanged=True,maximum_patch_mean_residual=maximum_mean_residual,strict_projection_only=True,original_training_unchanged=True)')
s=s.replace('# Same trained checkpoint; only feature injection is switched off.', '# Same trained checkpoint; only the inject convolution uses strict FP32. No training changes.')
(T/'precision_shadow480.py').write_bytes(s.encode())
