"""Fixed RGB local contrast and single-local-band deletion; original five stay."""
import numpy as np
from phase70_math import square_sum,checked

CONTROLS={'RGB_local':None,'B_noB08':[1,2],'B_noB11':[0,2],'B_noB12':[0,1]}


def rgb_local(raw,valid):
    den=square_sum(valid,30)-valid
    if np.any(den[valid]<=0):raise ValueError('Valid center without image-valid neighbors')
    columns=[]
    for channel in [3,2,1]:
        band=raw[...,channel].astype(np.float64)
        mean=(square_sum(np.where(valid,band,0.),30)[valid]-band[valid])/den[valid]
        columns.append((band[valid]-mean)/(np.abs(mean)+1e-6))
    return checked(np.column_stack(columns))


def inputs(original,b,rgb,name):
    if name=='L3':return original
    if name=='B':extra=b
    elif name=='RGB_local':extra=rgb
    else:extra=b[:,CONTROLS[name]]
    return np.column_stack([original,extra])
