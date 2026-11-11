"""Generate isolated Phase470 runners from the audited Phase469 protocol."""
from pathlib import Path
T=Path(__file__).parent
def copy(name,new=None):
 s=(T/name).read_text(encoding='utf-8').replace('469','470')
 return s
def put(name,s):
 (T/name).write_bytes(s.encode('utf-8'))
s=copy('train_shadow469.py')
s=s.replace('from shadow_vfm470 import OcclusionAttenuation,install','from shadow_experts470 import ErrorExperts,install')
s=s.replace("branch.vfm=nn.ModuleList([OcclusionAttenuation(relative=ARM=='B') for _ in range(4)]).cuda()","branch.experts=ErrorExperts(routed=ARM=='B').cuda()")
s=s.replace('branch.vfm','branch.experts').replace("'vfm.'","'experts.'")
s=s.replace(".05*sum(m.auxiliary(y) for m in branch.experts)/4",".1*branch.experts.auxiliary(y,reference)")
s=s.replace('Only four new VFM adapters trainable; all original parameters and BN fixed','Only new decoder-feature experts trainable; all original parameters and BN fixed')
s=s.replace("dict(relative_shadow=ARM=='B',rank=4,blocks=[8,9,10,11],zero_init=True,route_aux_weight=.05,position='Native VFM QKV patch tokens only')","dict(routed=ARM=='B',experts=4,zero_init=True,route_aux_weight=.1,position='First decoder pyramid feature, quarter resolution')")
s=s.replace('shadow_vfm470.py','shadow_experts470.py')
put('train_shadow470.py',s)
s=copy('infer_shadow469.py')
start=s.index("        if any(k.startswith('vfm.')")
end=s.index('        self.branch.load_state_dict',start)
s=s[:start]+'''        if any(k.startswith('experts.') for k in branch_state):
            from shadow_experts470 import ErrorExperts,install
            self.branch.experts=ErrorExperts(routed=bool(branch_state['experts.routed_mode']))
            install(self.teacher.backbone,self.branch.experts)
'''+s[end:]
put('infer_shadow470.py',s)
for name in ['verify_shadow469.py','boundary_shadow469.py','monitor_shadow469.py']:
 put(name.replace('469','470'),copy(name))
s=copy('ablate_shadow469.py').replace('[setattr(m,"enabled",False) for m in network.branch.vfm]','setattr(network.branch.experts,"enabled",False)')
put('ablate_shadow470.py',s)
s=copy('ablate_relative469.py').replace('[setattr(m,"relative_enabled",False) for m in network.branch.vfm]','setattr(network.branch.experts,"routing_enabled",False)').replace('relative_off','routing_off').replace('RELATIVE_OFF','ROUTING_OFF')
put('ablate_routing470.py',s)
s=copy('collect_shadow469.py').replace('relative_off','routing_off')
put('collect_shadow470.py',s)
s=copy('compare_shadow469.py').replace('B_vfm_on_minus_off','B_experts_on_minus_off').replace('B_relative_on_minus_off','B_routing_on_minus_off').replace('RELATIVE_OFF','ROUTING_OFF')
put('compare_shadow470.py',s)
s=copy('accept_shadow469.py').replace("'vfm.'","'experts.'").replace('relative_mode','routed_mode')
put('accept_shadow470.py',s)
s=copy('run_shadow469.sh').replace('ablate_relative470','ablate_routing470')
put('run_shadow470.sh',s)
