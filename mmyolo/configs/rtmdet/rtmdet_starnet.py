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
env_cfg = dict(cudnn_benchmark=True)

# StarNet配置 - 基于源码正确计算输出通道
base_dim = 24  # 根据你的配置
depths = [1, 2, 6, 2]  # 根据你的配置

# 根据StarNet源码分析，真实输出通道是:
# 从源码看，每个stage的输出通道为 base_dim * 2^i
# StarNet各阶段输出通道为: [48, 96, 192]
starnet_channels = [base_dim * 2**(i+1) for i in range(3)]  # [48, 96, 192]

# 修改模型配置，只替换backbone
model = dict(
   backbone=dict(
        _delete_=True, # 将 _base_ 中关于 backbone 的字段删除
        type='mmdet.PyramidVisionTransformerV2', # 使用 mmdet 中的 PyramidVisionTransformerV2
        embed_dims=32,
        num_layers=[2, 2, 2, 2],
        out_indices =(1, 2, 3), #设置PyramidVisionTransformerv2输出的stage，这里设置为1,2,3，默认为(0,1,2,3)
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)),
        plugins=[
            dict(cfg=dict(type='SKAttention'),
                 stages=(False, False, False, True))
        ]
    ),
    # 确保neck接收的通道数与backbone输出匹配
    neck=dict(
        in_channels=starnet_channels,  # [48, 96, 192]
    ),
    # 确保bbox_head配置正确
    bbox_head=dict(
        head_module=dict(num_classes=num_classes)  # 更新类别数
    )
)

# 使用原配置的数据加载器和pipeline，只修改数据集相关的设置
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
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
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=50
    ),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 4,
        end=max_epochs,
        T_max=max_epochs * 3 // 4,
        by_epoch=True,
        convert_to_iter_based=True
    ),
]

# 优化器配置
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