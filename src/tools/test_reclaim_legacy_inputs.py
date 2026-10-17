"""Deletion-boundary tests; no data files created or deleted."""
from reclaim_legacy_inputs import selected
def main():
    pid='S2A_MSIL1C_20180430T111621_N0206_R037_T40XDR_20180430T132750'
    n=12533808
    assert selected('phase72_20261010/prepared/arrays/'+pid+'_rgb.npy',n)
    assert selected('phase72_20261010/prepared/arrays/'+pid+'_B.npy',n)
    for relative in ['../phase72/'+pid+'_rgb.npy','/phase72/'+pid+'_rgb.npy','phase200_20261017/'+pid+'_rgb.npy','phase119_20261015/'+pid+'_rgb.npy',
        'phase72/'+pid+'_pred.npy','phase72/'+pid+'_msre_source_spectral.npy','phase72/'+pid+'_mask.npy','phase72/C_decoder_final.safetensors','subscenes/'+pid+'.npy']:
        assert not selected(relative,n),relative
    assert not selected('phase72/'+pid+'_rgb.npy',100)
    print('Cleanup boundary passed: protects originals, models, scores, masks, small files and all Omni-era phases')
if __name__=='__main__':main()
