"""Parameter-free serial/parallel Cloud-Adapter and MsRE connection."""
def connect(msre, raw, cloud, block_index, height, width, mode):
    assert mode in ('serial', 'parallel')
    if mode == 'parallel' and msre.enabled and block_index in msre.layer_slots:
        return cloud + (msre(raw, block_index, height, width) - raw)
    return msre(cloud, block_index, height, width)

def self_test():
    import torch
    class Residual(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.weight=torch.nn.Parameter(torch.tensor(.5)); self.enabled=True; self.layer_slots={2:0}
        def forward(self,x,index,h,w):
            return x+self.weight*x if self.enabled and index in self.layer_slots else x
    m=Residual(); raw=torch.tensor([2.]); cloud=raw*3
    serial=connect(m,raw,cloud,2,1,1,'serial');parallel=connect(m,raw,cloud,2,1,1,'parallel')
    assert serial.item()==9 and parallel.item()==7
    parallel.backward();assert m.weight.grad.item()==2
    m.enabled=False
    assert torch.equal(connect(m,raw,cloud,2,1,1,'parallel'),cloud)
    m.enabled=True
    assert torch.equal(connect(m,raw,cloud,1,1,1,'parallel'),cloud)
    return {'passed':True,'serial_parallel_differ_as_expected':True,'parallel_gradient_uses_raw_feature':True,'disabled_and_noninjection_identity':True,'added_parameters':0}
