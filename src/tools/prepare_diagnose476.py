from pathlib import Path
T=Path(__file__).parent
s=(T/'diagnose_contrast472.py').read_text().replace('472','476').replace('branch.contrast','branch.material')
s=s.replace('step>=2000','step>0').replace('assert net.branch.material.strength.item()==.25','assert net.branch.material.chroma_mode.item()')
s=s.replace('values=[v.item() for v in net.branch.material.relative_changes];assert len(values)==4 and np.isfinite(values).all()','values=[net.branch.material.relative_change.item()];assert np.isfinite(values).all()')
s=s.replace('strength=.25,','').replace('stage_mean_relative_input_rms=values.mean(0).tolist()','mean_relative_detail_rms=float(values.mean())').replace('stage_median_relative_input_rms=np.median(values,0).tolist()','median_relative_detail_rms=float(np.median(values))')
s=s.replace("(R/'CONTRAST_DIAG.json').write_text", "result['kernel_abs_mean']=net.branch.material.kernel.detach().abs().mean().item()\nresult['temperature']=(torch.nn.functional.softplus(net.branch.material.log_temperature)+.01).item()\n(R/'CONTRAST_DIAG.json').write_text")
(T/'diagnose_material476.py').write_bytes(s.encode())
