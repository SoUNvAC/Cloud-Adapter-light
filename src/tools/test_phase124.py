"""Only meaningful evidence classification boundaries/NA/unrounded directions."""
from audit_phase124 import direction,ranking,evidence
def main():
    assert ranking(1)=='stable' and ranking(-1)=='stable'
    assert ranking(1+1e-12)=='improved' and ranking(-1-1e-12)=='declined'
    assert direction(1e-15)=='increase' and direction(-1e-15)=='decrease'
    assert direction(None)==ranking(None)=='NA'
    data=[]
    for gid,positive in [('positive',10),('no_positive',0)]:
        for setting,ap,recall,fp in [('R_05',.5,.7,.05),('R_cal',.5,.5,.02),('C122_05',.52,.2,.01),('C122_cal',.52,.4,.015),('C123_05',.53,.2,.01),('C123_cal',.53,.4,.015)]:
            data.append(dict(group_id=gid,setting=setting,products='S2A_T40XDR',positive=str(positive),negative='20',AP=str(ap) if positive else '',recall=str(recall) if positive else '',FPR=str(fp),threshold='.5'))
    result=evidence(data,'synthetic');assert len(result)==4
    pos=[v for v in result if v['positive']];neg=[v for v in result if not v['positive']]
    assert all(v['two_seed_AP_repeat_class']=='both_improved' and v['ordering_gain_recall_loss'] for v in pos)
    assert all(v['AP_delta_vs_R_pp'] is None and v['cal_recall_delta_vs_R_pp'] is None and v['cal_zero_recall']=='NA' and v['cal_FPR_delta_vs_R_pp'] is not None for v in neg)
    print('Phase124 +/-1 boundary, raw direction, NA and mixed tradeoff CPU tests passed; no bootstrap/model calls')
if __name__=='__main__':main()
