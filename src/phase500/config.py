"""Standalone mmengine/mmseg recipe; no historical config inheritance."""
import os

default_scope = 'mmseg'
custom_imports = dict(imports=['phase500.integration'], allow_failed_imports=False)
env_cfg = dict(cudnn_benchmark=False, mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
               dist_cfg=dict(backend='nccl'))
randomness = dict(seed=42901, deterministic=False)
log_processor = dict(by_epoch=False)
log_level = 'INFO'
visualizer = dict(type='SegLocalVisualizer', vis_backends=[dict(type='LocalVisBackend')], name='visualizer')

attn = dict(embed_dims=128, num_heads=4, attn_drop=0.0, proj_drop=0.0,
            dropout_layer=None, batch_first=True)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(type='SegDataPreProcessor', mean=[123.675,116.28,103.53],
                           std=[58.395,57.12,57.375], bgr_to_rgb=True, size=(512,512),
                           pad_val=0, seg_pad_val=255),
    backbone=dict(type='DinoCloud500', repo=os.environ.get('DINOV3_REPO'),
                  weights=os.environ.get('DINOV3_WEIGHTS'), variant='texture', init_seed=42901),
    decode_head=dict(
        type='Mask2FormerHead', in_channels=[384]*4, strides=[4,8,16,32], feat_channels=128,
        out_channels=128, num_classes=3, num_queries=25, num_transformer_feat_level=3,
        align_corners=False, enforce_decoder_input_project=False,
        pixel_decoder=dict(type='mmdet.BoundaryPixelDecoder500', num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32), act_cfg=dict(type='ReLU'),
            encoder=dict(num_layers=2, layer_cfg=dict(
                self_attn_cfg=dict(embed_dims=128, num_heads=4, num_levels=3, num_points=4,
                    im2col_step=64, dropout=0.0, batch_first=True, norm_cfg=None, init_cfg=None),
                ffn_cfg=dict(embed_dims=128, feedforward_channels=512, num_fcs=2,
                             ffn_drop=0.0, act_cfg=dict(type='ReLU', inplace=True))), init_cfg=None),
            positional_encoding=dict(num_feats=64, normalize=True), init_cfg=None),
        positional_encoding=dict(num_feats=64, normalize=True),
        transformer_decoder=dict(return_intermediate=True, num_layers=2,
            layer_cfg=dict(self_attn_cfg=attn, cross_attn_cfg=attn,
                ffn_cfg=dict(embed_dims=128, feedforward_channels=512, num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True), ffn_drop=0.0,
                    dropout_layer=None, add_identity=True)), init_cfg=None),
        loss_cls=dict(type='mmdet.CrossEntropyLoss', use_sigmoid=False, loss_weight=2.0,
                      reduction='mean', class_weight=[1.,1.,1.,0.1]),
        loss_mask=dict(type='mmdet.CrossEntropyLoss', use_sigmoid=True, reduction='mean', loss_weight=5.0),
        loss_dice=dict(type='mmdet.DiceLoss', use_sigmoid=True, activate=True, reduction='mean',
                       naive_dice=True, eps=1.0, loss_weight=5.0),
        train_cfg=dict(num_points=12544, oversample_ratio=3.0, importance_sample_ratio=0.75,
            assigner=dict(type='mmdet.HungarianAssigner', match_costs=[
                dict(type='mmdet.ClassificationCost', weight=2.0),
                dict(type='mmdet.CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
                dict(type='mmdet.DiceCost', weight=5.0, pred_act=True, eps=1.0)]),
            sampler=dict(type='mmdet.MaskPseudoSampler'))),
    train_cfg=dict(), test_cfg=dict(mode='whole'))

data_root = os.environ.get('CLOUDSEN_ROOT', 'data/cloudsen12_high_l1c')
train_pipeline = [dict(type='LoadImageFromFile'), dict(type='LoadAnnotations'),
                  dict(type='RandomCrop', crop_size=(512,512)), dict(type='RandomFlip', prob=0.5),
                  dict(type='PhotoMetricDistortion'), dict(type='PackSegInputs')]
eval_pipeline = [dict(type='LoadImageFromFile'), dict(type='Resize', scale=(512,512)),
                 dict(type='LoadAnnotations'), dict(type='PackSegInputs')]
train_dataloader = dict(batch_size=2, num_workers=4, persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(type='CloudSEN500', data_root=data_root,
        data_prefix=dict(img_path='img_dir/train', seg_map_path='ann_dir/train'), pipeline=train_pipeline))
val_dataloader = dict(batch_size=2, num_workers=4, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type='CloudSEN500', data_root=data_root,
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'), pipeline=eval_pipeline))
test_dataloader = dict(batch_size=2, num_workers=4, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type='CloudSEN500', data_root=data_root,
        data_prefix=dict(img_path='img_dir/test', seg_map_path='ann_dir/test'), pipeline=eval_pipeline))
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU','mDice','mFscore'])
test_evaluator = val_evaluator
optim_wrapper = dict(type='OptimWrapper', constructor='TrainableOnly500',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.05, eps=1e-8, betas=(0.9,0.999)),
    clip_grad=dict(max_norm=1.0, norm_type=2), paramwise_cfg=dict(norm_decay_mult=0.0))
param_scheduler = [dict(type='LinearLR', start_factor=0.001, begin=0, end=500, by_epoch=False),
                   dict(type='PolyLR', power=0.9, eta_min=0., begin=500, end=40000, by_epoch=False)]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=2000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000, max_keep_ckpts=3,
                    save_best='mIoU', rule='greater'), sampler_seed=dict(type='DistSamplerSeedHook'))
custom_hooks = [dict(type='Audit500')]
load_from = None
resume = False
