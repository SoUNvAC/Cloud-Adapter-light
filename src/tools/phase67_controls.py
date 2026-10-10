"""Three fixed input ablations; no feature search or confirmation selection."""
import numpy as np

# Combined columns: MsRE, Source, B08, B11, B12, B04, B03, B02.
CONTROLS = {
    'L2_RGB': dict(columns=[0,1,5,6,7],positive_features=2,nonshadow='msre'),
    'Source_spectral': dict(columns=[1,2,3,4],positive_features=1,nonshadow='source'),
    'MsRE_spectral': dict(columns=[0,2,3,4],positive_features=1,nonshadow='msre'),
}


def combined(old_five, rgb):
    old_five=np.asarray(old_five);rgb=np.asarray(rgb)
    if old_five.ndim!=2 or old_five.shape[1]!=5 or rgb.shape!=(len(old_five),3):
        raise ValueError('Feature/grid mismatch')
    x=np.column_stack([old_five,rgb]).astype(np.float32)
    if not np.isfinite(x).all():raise ValueError('Invalid feature observation')
    return x


def selected(x,name):
    if name not in CONTROLS:raise ValueError('Only the three preregistered controls')
    return np.asarray(x)[:,CONTROLS[name]['columns']]


def choose(rows):
    """Development-only selection, no thresholds or inputs can change."""
    if set(rows)!=set(CONTROLS)|{'L3'}:raise ValueError('Fixed four selection candidates')
    eligible=[n for n,r in rows.items() if r['macro_fpr']<=.01 and r['macro_ap'] is not None]
    if not eligible:return None
    best=max(rows[n]['macro_ap'] for n in eligible)
    tied=[n for n in eligible if best-rows[n]['macro_ap']<=1e-12]
    return min(tied,key=lambda n:(rows[n]['n_features'],n))
