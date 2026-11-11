"""Source-label constrained image-space shadow pairs; no validation fitting."""
import torch
import math

def paired_shadow(rgb,y,step):
    generator=torch.Generator(device=rgb.device).manual_seed(453000+step)
    out=rgb.clone();labels=y.clone();stats=[]
    h,w=y.shape[-2:]
    yy,xx=torch.meshgrid(torch.arange(h,device=rgb.device),torch.arange(w,device=rgb.device),indexing='ij')
    for i in range(len(rgb)):
        ground=y[i]==0;locations=ground.nonzero()
        if not len(locations):
            stats.append(dict(core=0,transition=0));continue
        rand=torch.rand(6,device=rgb.device,generator=generator)
        cy,cx=locations[int(rand[0]*len(locations))]
        rx=32+96*rand[1];ry=32+96*rand[2];angle=rand[3]*math.pi
        dx=xx-cx;dy=yy-cy
        radius=torch.sqrt(((dx*angle.cos()+dy*angle.sin())/rx)**2+((-dx*angle.sin()+dy*angle.cos())/ry)**2)
        alpha=((1-radius)/.25).clamp(0,1)*ground
        attenuation=.20+.25*rand[4]
        tint=(rand[5]-.5)*.04
        scale=torch.stack([attenuation+tint,attenuation,attenuation-tint])
        out[i]=rgb[i]*(1-scale[:,None,None]*alpha)
        core=ground&(radius<=.75);transition=ground&(radius>.75)&(radius<1)
        labels[i][core]=2;labels[i][transition]=255
        assert torch.equal(out[i][:,~ground],rgb[i][:,~ground])
        assert torch.equal(labels[i][~ground],y[i][~ground])
        stats.append(dict(core=int(core.sum()),transition=int(transition.sum()),attenuation=float(attenuation),rx=float(rx),ry=float(ry)))
    assert torch.isfinite(out).all() and out.min()>=0 and out.max()<=255
    return out,labels,dict(step=step,images=stats,space='RGB image space, not calibrated physical reflectance')
