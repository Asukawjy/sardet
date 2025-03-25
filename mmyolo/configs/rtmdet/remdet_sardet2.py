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

# 如需采用 COCO 预训练权重，可根据需要设置 load_from（也可以直接使用基准配置中的预训练权重）

#load_from = 'https://download.openmmlab.com/mmyolo/v3.0/rtmdet/rtmdet_s_syncbn_fast_8xb32-300e_coco/rtmdet_s_syncbn_fast_8xb32-300e_coco_20221230_182329-8a9e1443.pth'
model = dict(
    # 冻结前3个stage的权重（加速训练）
    backbone=dict(
        ##修改部分
       frozen_stages=3,
        plugins=[
            dict(cfg=dict(type='SEAttention'),
                 stages=(False, False, False, True))
        ]
),
    # 头部模块调整，同时增加损失函数部分内容
    bbox_head=dict(
        head_module=dict(num_classes=num_classes)  # 分类数传递
        #loss_cls=dict(
         #   type='mmdet.CrossEntropyLoss',
          #  use_sigmoid=True,
           # reduction='mean',
            #loss_weight=loss_cls_weight *
            #(num_classes / 6 * 3 / num_det_layers)),
        # 修改此处实现IoU损失函数的替换
       # loss_bbox=dict(
        #    type='IoULoss',
         #   iou_mode='innerciou',
          #  bbox_format='xywh',
           # eps=1e-7,
           # reduction='mean',
           # loss_weight=loss_bbox_weight * (3 / num_det_layers),
           # return_iou=True),
    )
)
# 数据加载器配置，假设训练和验证的标注文件分别为 train_coco.json 与 val_coco.json，
# 图片存放在 data_root 下的 images/ 文件夹中
#train_pipeline = [
#    dict(type='LoadImageFromFile'),  # 加载图片
#    dict(type='LoadAnnotations', with_bbox=True),  # 加载标签
#    dict(type='Resize', scale=(640, 640), keep_ratio=True),  # 调整大小
#    dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),  # 随机翻转
#    dict(type='PhotoMetricDistortion'),  # 颜色变换
#    dict(type='RandomCrop', crop_size=(300, 300), allow_negative_crop=True),  # 随机裁剪
#    dict(type='CutOut', n_holes=1, cutout_shape=(50, 50)),  # Cutout
#    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),  # 归一化
#    dict(type='Pad', size_divisor=32),  # 填充到 32 的倍数
#    dict(type='DefaultFormatBundle'),  # 数据格式转换
#    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])  # 数据收集
#]
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
#test_dataloader = dict(
#    batch_size=val_batch_size_per_gpu,
 #   num_workers=val_num_workers,
  #  dataset=dict(
   #     metainfo=metainfo,
    #    data_root=data_root,
     #   ann_file='annotations/test.json',
      #  data_prefix=dict(img='images/test/')
  #  )
#)
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
