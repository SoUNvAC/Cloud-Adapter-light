"""Deployable combined network: original frozen431 plus learned occlusion branch."""
import torch
from torch import nn
from shadow_branch450 import LocalOcclusionBranch
class Frozen431WithOcclusion(nn.Module):
    def __init__(self,teacher,branch_state):
        super().__init__();self.teacher=teacher.eval().requires_grad_(False);self.branch=LocalOcclusionBranch();
        
        if any(k.startswith('transport.') for k in branch_state):
            from shadow_transport471 import ContourTransport,install
            self.branch.transport=ContourTransport(normal=bool(branch_state['transport.normal_mode']))
            install(self.teacher.backbone,self.branch.transport)
        self.branch.load_state_dict(branch_state,strict=True);self.values={}
        self.teacher.backbone.spatial.register_forward_hook(lambda m,a,o:self.values.update(detail=o[1]))
        self.teacher.backbone.register_forward_hook(lambda m,a,o:self.values.update(features=o))
        self.register_buffer('mean',torch.tensor([123.675,116.28,103.53])[None,:,None,None],persistent=False)
        self.register_buffer('std',torch.tensor([58.395,57.12,57.375])[None,:,None,None],persistent=False)
    def train(self,mode=True):
        super().train(mode);self.teacher.eval();return self
    def forward(self,rgb):
        # RGB0..255, Bx3x512x512, no augmentation or color alignment.
        rgb=rgb.float()
        with torch.no_grad():
            scores=self.teacher.encode_decode((rgb-self.mean)/self.std,[dict(img_shape=(512,512),ori_shape=(512,512),pad_shape=(512,512),padding_size=[0,0,0,0]) for _ in range(len(rgb))]).clamp_min(1e-8)
            base=(scores/scores.sum(1,keepdim=True)).log()
        logits,delta,gate=self.branch(self.values['detail'],self.values['features'],rgb/255,base)
        return logits,base,delta,gate
