from pathlib import Path
T=Path(__file__).parent
s=(T/'ablate_conditional474.py').read_text()
s=s.replace('setattr(network.branch.decoder,"conditional_enabled",False)',"network.branch.decoder.cloud.register_forward_hook(lambda m,a,o:o.mean((2,3),keepdim=True).expand_as(o));network.branch.decoder.shadow.register_forward_hook(lambda m,a,o:o.mean((2,3),keepdim=True).expand_as(o))")
assert 'conditional_enabled",False' not in s
s=s.replace('conditional_off','spatial_mean').replace('CONDITIONAL_OFF','SPATIAL_MEAN')
(T/'ablate_spatial474.py').write_bytes(s.encode())
p=T/'collect_shadow474.py';s=p.read_text().replace("'conditional_off'","'conditional_off','spatial_mean'");p.write_bytes(s.encode())
p=T/'compare_shadow474.py';s=p.read_text();last=s[s.index("cls=json.loads"):].replace('CONDITIONAL_OFF','SPATIAL_MEAN').replace('B_conditional_on_minus_off','B_spatial_minus_mean');p.write_bytes((s+'\n'+last).encode())
s=(T/'verify_archive473.py').read_text().replace('473','474');(T/'verify_archive474.py').write_bytes(s.encode())
