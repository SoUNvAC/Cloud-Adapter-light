"""Three-class fixed-decision metrics, with explicit per-reference denominators."""
import numpy as np
def metrics(c):
 c=np.asarray(c,dtype=np.int64);diag=np.diag(c);u=c.sum(0)+c.sum(1)-diag;iou=np.divide(diag,u,out=np.full(3,np.nan),where=u>0)*100
 def ratio(n,d):return float(100*n/d) if d else None
 return {'mIoU':float(np.nanmean(iou)),'class_IoU':[float(x) if np.isfinite(x) else None for x in iou],'valid_classes':int((u>0).sum()),'Cloud_recall':ratio(c[1,1],c[1].sum()),'Surface_to_Cloud_FPR':ratio(c[0,1],c[0].sum()),'Shadow_recall':ratio(c[2,2],c[2].sum()),'nonShadow_to_Shadow_FPR':ratio(c[:2,2].sum(),c[:2].sum()),'Surface_pixels':int(c[0].sum()),'Cloud_pixels':int(c[1].sum()),'Shadow_pixels':int(c[2].sum()),'nonShadow_pixels':int(c[:2].sum()),'confusion':c.tolist()}
def summary(per):
 pooled=metrics(sum(np.asarray(x['confusion'],dtype=np.int64) for x in per));macro={};n={}
 for k in ['mIoU','Cloud_recall','Surface_to_Cloud_FPR','Shadow_recall','nonShadow_to_Shadow_FPR']:
  a=[v[k] for v in per if v[k] is not None];macro[k]=float(np.mean(a)) if a else None;n[k]=len(a)
 return {'pooled':pooled,'scene_mean':macro,'defined_scenes':n}
