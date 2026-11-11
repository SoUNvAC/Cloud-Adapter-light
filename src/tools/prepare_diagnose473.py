from pathlib import Path
T=Path(__file__).parent
s=(T/'diagnose_contrast472.py').read_text().replace('472','473')
s=s.replace("per.append(dict(index=row['index'],group=row['group'],relative_input_rms=values))",'''raw=np.array(Image.open(row['mask']));valid=raw!=255;truth=raw.copy()
 for k,v in {0:0,1:1,2:1,3:2}.items():truth[raw==k]=v
 mass=torch.from_numpy(np.stack([(truth==c)&valid for c in range(3)]).astype(np.float32))[None].cuda()
 statistics=[]
 for cls in net.branch.contrast.class_logits:
  target=torch.nn.functional.adaptive_avg_pool2d(mass,cls.shape[-2:]);prob=cls.softmax(1)
  statistics.append(dict(mass=target.sum((0,2,3)).cpu().tolist(),sum_probability=torch.einsum('bchw,bjhw->cj',target,prob).cpu().tolist()))
 per.append(dict(index=row['index'],group=row['group'],relative_input_rms=values,class_probability=statistics))''')
s=s.replace("(R/'CONTRAST_DIAG.json').write_text",'''sums=np.array([[v['sum_probability'] for v in p['class_probability']] for p in per]).sum(0)
mass=np.array([[v['mass'] for v in p['class_probability']] for p in per]).sum(0)
result['stage_true_class_mean_predicted_probability']=(sums/mass[:,:,None]).tolist()
(R/'CONTRAST_DIAG.json').write_text''')
(T/'diagnose_contrast473.py').write_bytes(s.encode())
