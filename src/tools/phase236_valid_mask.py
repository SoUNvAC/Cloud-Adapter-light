"""Run-local Mask2Former correction; no installed package is changed.

The all-valid branch calls the unmodified installed implementation. With Fill,
both matching and uncertain-point sampling draw valid GT pixel centres. On both
grids, interpolation excludes invalid nodes and renormalizes the remaining
weights. This preserves supervision even for a single valid pixel.
"""
import copy, hashlib, inspect, textwrap, types, re


def install(head):
    import torch
    import torch.nn.functional as F
    import mmdet.models.dense_heads.mask2former_head as mod
    from mmcv.ops import point_sample

    original_seg = head._seg_data_to_instance_data
    original_target = head._get_targets_single
    original_loss = head._loss_by_feat_single
    head._phase236_generator = torch.Generator(device=next(head.parameters()).device)
    head._phase236_generator.manual_seed(236)
    head._phase236_sampling = []

    def seg(self, samples):
        instances, metas = original_seg(samples)
        metas = [dict(m) for m in metas]
        for sample, meta in zip(samples, metas):
            meta['_phase236_valid'] = sample.gt_sem_seg.data.squeeze(0) != self.ignore_index
        return instances, metas

    def grids(valid, pred):
        v = valid[None, None].float().to(pred.device)
        # Invalid pooling bins exclude prediction nodes conservatively. For the
        # synthetic test and runtime native grids this includes all invalid cells.
        p = F.adaptive_max_pool2d(v, pred.shape[-2:])
        return v, p

    def allowed(coords, valid, pred):
        vg, pg = grids(valid, pred)
        n = coords.shape[0]
        a = point_sample(vg.expand(n,-1,-1,-1), coords).squeeze(1)
        b = point_sample(pg.expand(n,-1,-1,-1), coords).squeeze(1)
        return (a >= 1-1e-5) & (b > 0)

    def sample_valid(valid, pred, n, count):
        # Independent RNG: the new rejection draws cannot perturb model/dropout
        # RNG or the historical sampler/augmentation sequence.
        ys,xs=torch.where(valid.to(pred.device))
        assert len(ys)>0,'All-invalid crop must have been excluded before sampling'
        index=torch.randint(len(ys),(n,count),device=pred.device,generator=head._phase236_generator)
        coords=torch.stack(((xs[index].float()+.5)/valid.shape[-1],(ys[index].float()+.5)/valid.shape[-2]),-1)
        assert allowed(coords,valid,pred).all()
        head._phase236_sampling.append({'points':n*count,'valid_GT_centres':True,'invalid_interpolation_weights_zeroed_and_renormalized':True})
        return coords

    def matching_points(valid, pred):
        head._phase236_active_valid=valid
        if valid is None or bool(valid.all()):
            return torch.rand((1,head.num_points,2),device=pred.device)
        return sample_valid(valid,pred,1,head.num_points)

    def uncertainty_points(pred, labels, count, over, importance, valid):
        if valid is None or bool(valid.all()):
            return mod.get_uncertain_point_coords_with_randomness(pred,labels,count,over,importance)
        n=pred.shape[0]; candidates=sample_valid(valid,pred,n,int(count*over))
        values=safe_point_sample(pred,candidates).squeeze(1)
        k=int(importance*count)
        idx=torch.topk(-values.abs(),k,dim=1).indices
        top=candidates.gather(1,idx[:,:,None].expand(-1,-1,2))
        rest=sample_valid(valid,pred,n,count-k)
        return torch.cat((top,rest),1)

    def safe_point_sample(input,coords,**kwargs):
        valid=getattr(head,'_phase236_active_valid',None)
        if valid is None or bool(valid.all()):return point_sample(input,coords,**kwargs)
        # A prediction node is eligible iff its GT footprint contains a valid
        # pixel. Invalid nodes have exactly zero weight and zero gradient.
        vg,pg=grids(valid,input)
        weights=pg.expand(input.shape[0],1,-1,-1)
        denominator=point_sample(weights,coords,**kwargs)
        assert (denominator>0).all()
        return point_sample(input*weights,coords,**kwargs)/denominator

    target_source=textwrap.dedent(inspect.getsource(original_target.__func__))
    pattern=r'point_coords = torch\.rand\(\(1, self\.num_points, 2\),\s*device=cls_score\.device\)'
    assert len(re.findall(pattern,target_source))==1,'Installed matching implementation drift'
    patched_target=re.sub(pattern,"point_coords = matching_points(img_meta.get('_phase236_valid'), mask_pred)",target_source)
    loss_source=textwrap.dedent(inspect.getsource(original_loss.__func__))
    needle2='self.oversample_ratio, self.importance_sample_ratio)'
    assert loss_source.count(needle2)==1 and loss_source.count('get_uncertain_point_coords_with_randomness(')==1
    patched_loss=loss_source.replace('get_uncertain_point_coords_with_randomness(', 'uncertainty_points(').replace(needle2,"self.oversample_ratio, self.importance_sample_ratio,\n                batch_img_metas[0].get('_phase236_valid'))")
    # All prescribed fits are batch1; passing batch0 validity to matched masks
    # is therefore exact. Fail rather than silently handle a larger batch.
    patched_loss=patched_loss.replace('num_imgs = cls_scores.size(0)','num_imgs = cls_scores.size(0)\n    assert num_imgs == 1')
    ns=dict(mod.__dict__);ns.update(matching_points=matching_points,uncertainty_points=uncertainty_points,point_sample=safe_point_sample)
    exec(patched_target,ns); target=ns['_get_targets_single']
    exec(patched_loss,ns); loss=ns['_loss_by_feat_single']
    head._seg_data_to_instance_data=types.MethodType(seg,head)
    head._get_targets_single=types.MethodType(target,head)
    head._loss_by_feat_single=types.MethodType(loss,head)
    return {'original_seg':inspect.getsource(original_seg.__func__),
            'original_target':target_source,'patched_target':patched_target,
            'original_loss':loss_source,'patched_loss':patched_loss,
            'all_valid_exact_original':True,'batch_size':1,
            'interpolation_policy':'valid GT pixel centres; prediction/GT invalid nodes zero-weighted with normalized bilinear interpolation, any-valid footprint prediction nodes; supports singleton valid pixel',
            'new_sampling_RNG':'independent torch.Generator, seeded per fixed training step',
            'original_single_loss':original_loss,'prediction_valid_grid':grids}
