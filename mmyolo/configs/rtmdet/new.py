_base_ = 'remdet_sardet2.py'  # 替换为你原始的配置文件路径

# 定义继续训练所需的变量
base_lr = 12 * 0.004 / (32 * 8)  # 确保与原配置文件中的定义相同

# 加载最后的检查点
load_from = 'work_dirs/remdet_sardet2/epoch_300.pth'  # 替换为你的检查点文件路径

# 设置新的训练轮次
max_epochs = 450  # 设置新的总轮次
num_epochs_stage2 = 10  # 最后10轮使用第二阶段pipeline

# 设置从第300轮继续训练
# 注意：mmyolo使用的是mmengine，可能需要不同的初始化方式
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=5,
    _base_epoch=300  # 从第300轮开始
)

# 调整学习率策略
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=300,  # 从第300轮开始
        end=max_epochs,
        T_max=max_epochs - 300,
        by_epoch=True,
        convert_to_iter_based=True
    ),
]

# 确保检查点保存策略不会覆盖之前的重要检查点
default_hooks = dict(
    checkpoint=dict(interval=5, max_keep_ckpts=5, save_best='auto'),
    logger=dict(type='LoggerHook', interval=5)
)

# 设置第二阶段pipeline切换点
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - num_epochs_stage2,
        switch_pipeline='train_pipeline_stage2')  # 确保这个pipeline已定义
]