from pathlib import Path
T=Path(__file__).parent
s=(T/'ablate_appearance473.py').read_text()
s=s.replace('setattr(network.branch.contrast,"appearance_enabled",False)',"network.branch.contrast.classes.register_forward_hook(lambda m,a,o:o.softmax(1).mean((2,3),keepdim=True).clamp_min(1e-8).log().expand_as(o))")
s=s.replace('appearance_off','uniform_routing').replace('APPEARANCE_OFF','UNIFORM_ROUTING')
(T/'ablate_uniform473.py').write_bytes(s.encode())
p=T/'collect_shadow473.py';s=p.read_text().replace("'appearance_off'","'appearance_off','uniform_routing'");p.write_bytes(s.encode())
