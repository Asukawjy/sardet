_base_ = 'rtmdet_s_syncbn_fast_8xb32-300e_coco.py'

data_root = 'autodl-tmp/mmyolo/data/SmallDataset_2/'

# 定义数据集的类别信息及颜色调色板
metainfo = {
    'classes': ("ship", "aircraft", "car", "tank", "bridge", "harbor"),
    'palette': [
        (0, 0, 255),      # ship: 蓝色
        (255, 0, 0),      # aircraft: 红色
        (0, 255, 0),      # car: 绿色
        (255, 255, 0),    # tank: 黄色
        (0, 255, 255),    # bridge: 青色
        (255, 0, 255)     # harbor: 品红
    ]
}
num_classes = 6
deepen_factor = _base_.deepen_factor
widen_factor = 1.0
# 修改通道设置方式
channels_in = [64, 160, 256]
channel_out = 256  # 设定一个固定的输出通道数，用于 neck 的 out_channels
checkpoint_file = 'https://bgithub.xyz/whai362/PVT/releases/download/v2/pvt_v2_b0.pth'
# 训练配置
max_epochs = 300
train_batch_size_per_gpu = 12
train_num_workers = 4

# 验证配置
val_batch_size_per_gpu = 1
val_num_workers = 2

# RTMDet 训练过程分为两个阶段，第二阶段切换数据增强 pipeline 的 epoch
num_epochs_stage2 = 5

# 根据单卡训练的 batch 大小计算学习率
base_lr = 12 * 0.004 / (32 * 8)
num_det_layers = 3

model = dict(
    backbone=dict(
        ## 修改部分
        _delete_=True, # 将 _base_ 中关于 backbone 的字段删除
        type='mmdet.PyramidVisionTransformerV2', # 使用 mmdet 中的 PyramidVisionTransformerV2
        embed_dims=32,
        num_layers=[2, 2, 2, 2],
        out_indices=(1, 2, 3), #设置PyramidVisionTransformerv2输出的stage，这里设置为1,2,3
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
    ),
    neck=dict(
        # CSPNeXtPAFPN 需要单独的 in_channels 和 out_channels
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=channels_in,  # 输入通道列表
        out_channels=channel_out,  # 固定的输出通道数
        dict(
            type='mmdet.ChannelMapper',
            in_channels=[256, 256, 256],  # 对应CSPNeXtPAFPN的三个输出
            out_channels=256,
        ),
        # 自定义neck，包含SKAttention
        dict(
            type='SKAdapter',
            in_channels=256,
            reduction=8,
        )
    ),
    bbox_head=dict(
        head_module=dict(
            num_classes=num_classes,
            # 由于 neck 输出的通道数现在是 channel_out，所以 head 的输入通道也要相应调整
            in_channels=channel_out,
            widen_factor=widen_factor)#,
        #loss_bbox=dict(
        #    type='IoULoss',
        #    iou_mode='eiou',
        #    bbox_format='xywh',
        #    eps=1e-7,
        #    reduction='mean',
        #    loss_weight=3.0,  # 略微增加回归损失权重
        #    return_iou=False
        #)
    )
)
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    pin_memory=False,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/train/')
    )
)

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/val.json',
        data_prefix=dict(img='images/val/')
    )
)

test_dataloader = val_dataloader

# 学习率调度器设置
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=30
    ),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True
    ),
]

optim_wrapper = dict(optimizer=dict(lr=base_lr))

# 切换 pipeline 的 epoch 时刻（对应第二阶段）
_base_.custom_hooks[1].switch_epoch = max_epochs - num_epochs_stage2

val_evaluator = dict(ann_file=data_root + 'annotations/val.json')
test_evaluator = val_evaluator

# 打印和检查点相关的设置
default_hooks = dict(
    checkpoint=dict(interval=5, max_keep_ckpts=2, save_best='auto'),
    logger=dict(type='LoggerHook', interval=5)
)
train_cfg = dict(max_epochs=max_epochs, val_interval=5)