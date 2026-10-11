"""T=1, lambda=1 conditional Surface/Cloud KL on nonshadow GT pixels."""
import torch
import torch.nn.functional as F

def conditional_kl(student,teacher,labels):
    assert student.shape==teacher.shape and student.shape[1]==3
    assert student.shape[0]==labels.shape[0] and student.shape[-2:]==labels.shape[-2:]
    assert student.dtype==teacher.dtype==torch.float32
    log_s=F.log_softmax(student[:,:2],dim=1)
    log_t=F.log_softmax(teacher.detach()[:,:2],dim=1)
    pixel=(log_t.exp()*(log_t-log_s)).sum(1)
    valid=(labels==0)|(labels==1)
    return pixel[valid].mean() if valid.any() else student[:,:2].sum()*0

def register_student():
    from mmseg.registry import MODELS
    from cloud_adapter.models.segmentors.frozen_encoder_decoder import FrozenHeadEncoderDecoder

    @MODELS.register_module()
    class Phase82ConditionalKDStudent(FrozenHeadEncoderDecoder):
        def loss(self,inputs,data_samples):
            # Teacher is deliberately not a child module: excluded from optimizer/checkpoint.
            teacher=getattr(self,'_phase82_teacher')
            assert not teacher.training and not any(p.requires_grad for p in teacher.parameters())
            metas=[s.metainfo for s in data_samples]
            with torch.no_grad():
                teacher_logits=teacher.decode_head.predict(teacher.extract_feat(inputs),metas,teacher.test_cfg).float()
            x=self.extract_feat(inputs)
            losses=self._decode_head_forward_train(x,data_samples)
            if self.with_auxiliary_head:losses.update(self._auxiliary_head_forward_train(x,data_samples))
            student_logits=self.decode_head.predict(x,metas,self.test_cfg).float()
            labels=torch.stack([s.gt_sem_seg.data for s in data_samples]).squeeze(1)
            shape=labels.shape[-2:]
            if student_logits.shape[-2:]!=shape:
                student_logits=F.interpolate(student_logits,size=shape,mode='bilinear',align_corners=self.align_corners)
            if teacher_logits.shape[-2:]!=shape:
                teacher_logits=F.interpolate(teacher_logits,size=shape,mode='bilinear',align_corners=self.align_corners)
            kd=conditional_kl(student_logits,teacher_logits,labels)
            losses['distill.loss_kd']=kd
            for key,value in losses.items():
                for v in (value if isinstance(value,list) else [value]):
                    assert torch.isfinite(v).all(),('Nonfinite loss; stop without retuning',key)
            self._phase82_last_losses=dict(kd=float(kd.detach()),
                supervised=float(sum((sum(v) if isinstance(v,list) else v).mean() for k,v in losses.items() if 'loss' in k and k!='distill.loss_kd').detach()),
                valid_nonshadow_pixels=int(((labels==0)|(labels==1)).sum()),
                total_label_pixels=int(labels.numel()),same_augmented_input_tensor=True)
            return losses
    return Phase82ConditionalKDStudent

def torch_preflight():
    # Hand-reference probabilities 0.8/0.2 vs 0.5/0.5; Shadow and ignore excluded.
    import math
    t=torch.tensor([[[[math.log(.8),0,0]],[[math.log(.2),0,0]],[[100.,0,0]]]],dtype=torch.float32)
    s=torch.zeros_like(t,requires_grad=True);labels=torch.tensor([[[0,2,255]]])
    got=conditional_kl(s,t,labels);expected=.8*math.log(1.6)+.2*math.log(.4)
    assert abs(float(got)-expected)<1e-6
    got.backward();assert s.grad[0,2].abs().sum()==0 and s.grad[:,:,:,1:].abs().sum()==0
    assert abs(float(conditional_kl(t,t,labels)))<1e-7
    e=s.detach().clone().requires_grad_();zero=conditional_kl(e,t,torch.full_like(labels,255));zero.backward()
    assert float(zero)==0 and e.grad.abs().sum()==0
    return dict(hand_KL=float(got),expected=expected,empty_mask_zero=True,same_distribution_zero=True,
                shadow_logit_excluded=True,shadow_and_ignore_gradients_zero=True)
