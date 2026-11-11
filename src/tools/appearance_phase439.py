"""Label-independent interventions on linear TOA RGB*255; no display gamma."""
import numpy as np
from scipy.ndimage import gaussian_filter


def conditions():
    result=[dict(id='original',family='original',value=0.,tier='reference')]
    for family,values,mild in [
        ('exposure',[-1.,-.5,.5,1.],[-.5,.5]),
        ('contrast',[.6,.8,1.2,1.4],[.8,1.2]),
        ('red_blue',[.5,.7,1.4,2.],[.7,1.4]),
        ('sharpen',[.5,1.,2.],[.5])]:
        for value in values:
            result.append(dict(id=f'{family}_{value:g}',family=family,value=value,
                               tier='mild' if value in mild else 'stress'))
    return result


def transform(rgb,image_valid,condition):
    """Return transformed input, metadata, and newly clipped pixel mask."""
    assert rgb.dtype==np.float32 and rgb.ndim==3 and rgb.shape[-1]==3
    assert image_valid.shape==rgb.shape[:2] and image_valid.any()
    assert np.isfinite(rgb).all() and (rgb[image_valid]>=0).all()
    family=condition['family'];value=condition['value']
    x=rgb.astype(np.float64)
    center=float(x[image_valid].mean())
    meta=dict(brightness_center=center,upper_clipping_enabled=False,
              luminance_definition='arithmetic RGB mean; sensor-band brightness proxy, not calibrated CIE luminance')
    if family=='original':
        raw=x
    elif family=='exposure':
        raw=x*np.exp2(value)
        meta['exposure_factor']=float(np.exp2(value))
    elif family=='contrast':
        raw=center+value*(x-center)
    elif family=='red_blue':
        gains=np.array([np.sqrt(value),1.,1./np.sqrt(value)],np.float64)
        preliminary=x*gains
        compensation=center/float(preliminary[image_valid].mean())
        gains*=compensation
        raw=x*gains
        meta.update(gains=gains.tolist(),red_blue_gain_ratio=float(gains[0]/gains[2]),
                    brightness_compensation=compensation)
    elif family=='sharpen':
        # Fixed sigma 2 native pixels (~1 model-input pixel), 13x13 support.
        # Normalized convolution prevents invalid zero borders bleeding into blur.
        weight=gaussian_filter(image_valid.astype(np.float64),2.,mode='reflect',truncate=3.)
        blurred=gaussian_filter(x*image_valid[...,None],(2.,2.,0.),mode='reflect',truncate=3.)
        blurred/=np.maximum(weight[...,None],1e-12)
        raw=x+value*(x-blurred)
        meta.update(sigma_native_pixels=2.,kernel_size=13,unsharp_amount=value)
    else:
        raise ValueError(family)
    low=(raw<0).any(-1)&image_valid
    newly_zero=((raw<=0)&(x>0)).any(-1)&image_valid
    output=np.maximum(raw,0).astype(np.float32)
    output[~image_valid]=0
    assert np.isfinite(output).all()
    if family=='original':assert np.array_equal(output,rgb)
    baseline_above=(x>255).any(-1)
    above=(output>255).any(-1)
    count=int(image_valid.sum())
    meta.update(image_valid_pixels=count,lower_clipped_pixels=int(low.sum()),
                newly_zero_pixels=int(newly_zero.sum()),upper_clipped_pixels=0,
                baseline_above_reflectance1_pixels=int((baseline_above&image_valid).sum()),
                above_reflectance1_pixels=int((above&image_valid).sum()),
                newly_above_reflectance1_pixels=int((above&~baseline_above&image_valid).sum()),
                mean_brightness_before=center,mean_brightness_after=float(output[image_valid].mean()),
                min_before=float(x[image_valid].min()),max_before=float(x[image_valid].max()),
                min_after=float(output[image_valid].min()),max_after=float(output[image_valid].max()))
    return output,meta,newly_zero


def self_test():
    x=np.array([[[10,20,30],[80,100,120]],[[200,300,400],[0,0,0]]],dtype=np.float32)
    valid=np.array([[1,1],[1,0]],bool)
    y,_,_=transform(x,valid,dict(family='original',value=0))
    assert np.array_equal(x,y)
    for family,neutral in [('exposure',0),('contrast',1),('red_blue',1),('sharpen',0)]:
        y,_,_=transform(x,valid,dict(family=family,value=neutral))
        assert np.array_equal(x,y),(family,'identity')
    y,_,_=transform(x,valid,dict(family='exposure',value=1))
    assert np.array_equal(y,2*x) and y.max()>255
    for value in [.5,.7,1.4,2.]:
        y,m,_=transform(x,valid,dict(family='red_blue',value=value))
        assert abs(m['red_blue_gain_ratio']-value)<1e-12
        assert np.isclose(y[valid].mean(),x[valid].mean(),rtol=1e-6)
    y,m,clip=transform(x,valid,dict(family='contrast',value=1.4))
    assert clip.any() and m['upper_clipped_pixels']==0 and (y>=0).all()
    assert len(conditions())==16
    return dict(passed=True,identities=5,linear_exposure_verified=True,white_balance_ratio_verified=True,
                no_upper_clipping=True,invalid_pixels_preserved=True)
