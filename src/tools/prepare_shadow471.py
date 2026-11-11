from pathlib import Path
T=Path(__file__).parent
def put(n,s):(T/n).write_bytes(s.encode('utf-8'))
def copy(n):return (T/n).read_text(encoding='utf-8').replace('470','471')
s=copy('train_shadow470.py').replace('shadow_experts471','shadow_transport471').replace('ErrorExperts','ContourTransport').replace("routed=ARM=='B'","normal=ARM=='B'").replace('branch.experts','branch.transport').replace("'experts.'","'transport.'")
s=s.replace('new decoder-feature experts','new contour transport layer').replace("dict(normal=ARM=='B',experts=4,zero_init=True,route_aux_weight=.1,position='First decoder pyramid feature, quarter resolution')","dict(normal=ARM=='B',max_flow_grid=2,zero_init=True,distance_aux_weight=.1,position='First decoder pyramid feature, quarter resolution')")
put('train_shadow471.py',s)
s=copy('infer_shadow470.py').replace('shadow_experts471','shadow_transport471').replace('ErrorExperts','ContourTransport').replace('experts.','transport.').replace('branch.experts','branch.transport').replace('routed=','normal=').replace('routed_mode','normal_mode')
put('infer_shadow471.py',s)
for n in ['verify_shadow470.py','boundary_shadow470.py','monitor_shadow470.py']:
 put(n.replace('470','471'),copy(n))
put('ablate_shadow471.py',copy('ablate_shadow470.py').replace('branch.experts','branch.transport'))
s=copy('ablate_routing470.py').replace('branch.experts','branch.transport').replace('routing_enabled','normal_enabled').replace('routing_off','normal_off').replace('ROUTING_OFF','NORMAL_OFF')
put('ablate_normal471.py',s)
put('collect_shadow471.py',copy('collect_shadow470.py').replace('routing_off','normal_off'))
s=copy('compare_shadow470.py').replace('B_experts_on_minus_off','B_transport_on_minus_off').replace('B_routing_on_minus_off','B_normal_on_minus_off').replace('ROUTING_OFF','NORMAL_OFF')
put('compare_shadow471.py',s)
put('accept_shadow471.py',copy('accept_shadow470.py').replace("'experts.'","'transport.'").replace('routed_mode','normal_mode'))
put('run_shadow471.sh',copy('run_shadow470.sh').replace('ablate_routing471','ablate_normal471'))
