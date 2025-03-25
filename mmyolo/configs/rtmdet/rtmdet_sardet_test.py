# 基础配置继承（保持原有配置）
_base_ = 'rtmdet_s_syncbn_fast_8xb32-300e_coco.py'

# ------------------ 数据集配置 ------------------
metainfo = {
    'classes': ("ship", "aircraft", "car", "tank", "bridge", "harbor"),
    'palette': [
        (0, 0, 255), (255, 0, 0), (0, 255, 0),
        (255, 255, 0), (0, 255, 255), (255, 0, 255)
    ]
}
num_classes = 6
data_root = 'autodl-tmp/mmyolo/data/SmallDataset_2/'

# ------------------ 模型结构优化 ------------------
model = dict(
    # 冻结前2层（平衡速度与特征学习）
    backbone=dict(frozen_stages=2),
    
    # 检测头优化
    bbox_head=dict(
        # 锚框尺寸适配遥感目标
        anchor_generator=dict(
            scales_per_octave=3,
            ratios=[1.0, 2.5, 5.0],  # 长条形目标适配
            strides=[8, 16, 32],
            centers=[(4, 4), (8, 8), (16, 16)]
        ),
        
        # 损失函数定制
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=3.0,  # 加强难样本挖掘
            alpha=0.75, # 缓解类别不平衡
            loss_weight=0.15  # 原始计算值基础上提升2倍
        ),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='innerciou',  # 处理密集目标重叠
            loss_weight=1.5        # 降低回归权重
        ),
        loss_obj=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=3.0        # 增强目标存在性判断
        ),
        
        # 检测头参数调整
        head_module=dict(
            num_classes=num_classes,
            featmap_strides=[8, 16, 32],
            in_channels=256,
            widen_factor=0.75      # 减少通道数加速推理
        )
    )
)

# ------------------ 数据增强策略 ------------------
# 针对低分辨率与小目标的增强
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations', with_bbox=True),
    # 去噪增强
    dict(type='PhotoMetricDistortion',
         brightness_delta=32,
         contrast_range=(0.8, 1.2),
         saturation_range=(0.8, 1.2)),
    # 几何增强
    dict(type='RandomRotate', prob=0.6, max_rot_angle=45),
    dict(type='RandomFlip', prob=0.5),
    # 多尺度训练
    dict(type='RandomResize',
         scale=(640, 1920),  # 宽高动态调整
         ratio_range=(0.8, 1.2),
         keep_ratio=True),
    dict(type='Pad', size_divisor=32),
    dict(type='PackDetInputs')
]

# ------------------ 训练策略优化 ------------------
# 学习率调整（小数据集适用）
base_lr = 0.004 * 12 / (32 * 8)  # 原基础学习率
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=base_lr * 0.8, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bypass_duplicate=True,
        custom_keys={
            'backbone': dict(lr_mult=0.1),  # 骨干网络更低学习率
            'neck': dict(lr_mult=0.3),
            'bbox_head': dict(lr_mult=1.0)
        }))

# 训练轮次与调度
max_epochs = 400
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-5, by_epoch=False, begin=0, end=50),
    dict(type='CosineAnnealingLR',
         T_max=max_epochs,  # 全周期余弦衰减
         eta_min=base_lr * 0.01,
         begin=50,
         end=max_epochs)
]

# ------------------ 推理优化 ------------------
# 后处理参数调整
test_cfg = dict(
    nms_pre=1000,    # 提高召回率
    score_thr=0.01,  # 降低初始阈值
    nms=dict(type='soft_nms', iou_threshold=0.5),  # 软NMS处理密集目标
    max_per_img=100  # 允许更多检测结果
)

# ------------------ 其他关键配置 ------------------
# 验证频率
train_cfg = dict(max_epochs=max_epochs, val_interval=3)
# 加速配置
env_cfg = dict(cudnn_benchmark=True)