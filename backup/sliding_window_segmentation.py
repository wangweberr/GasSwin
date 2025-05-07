# 配置文件路径：configs/segmentation/sliding_window_segmentation.py
# 用于 180×200 分辨率、9 通道、滑窗输入的视频分割任务

# ===================== 模型设置 =====================
model = dict(
    type='Segmentor',  # 使用自定义的 Segmentor 识别器
    backbone=dict(
        type='SwinTransformer3D',
        pretrained=None,  # 或者填写预训练权重路径
        # 以下参数根据实际模型调整
        patch_size=(2,4,4),
        depths=[2,2,6,2],
        num_heads=[3,6,12,24],
        embed_dim=96,
        drop_path_rate=0.2
    ),
    seg_head=dict(
        type='DynamicSegmentationHead',
        in_channels=384,   # 根据 backbone 最后一层输出通道数
        num_classes=2,
        loss_seg=dict(type='CrossEntropyLoss', loss_weight=1.0)
    ),
    test_cfg=dict(mode='whole')
)

# ===================== 数据集设置 =====================
dataset_type = 'SlidingWindowRawframeSegmentationDataset'
data_root = 'data/your_dataset'  # 修改为你的数据根目录
window_size = 32
stride = 16
filename_tmpl = '{:05d}.png'
# 二分类示例，可按需修改
num_classes = 2

# 图像归一化配置（9 通道）
img_norm_cfg = dict(
    mean=[0] * 9,
    std=[1] * 9,
    to_bgr=False
)

# ================ 训练/验证/测试数据流 ====================
train_pipeline = [
    dict(
        type='SlidingWindowSampler',
        window_size=window_size,
        stride=stride,
        test_mode=False
    ),
    dict(type='RawFrameDecode', io_backend='disk'),
    dict(type='LoadAnnotations', with_seg=True, seg_prefix=f'{data_root}/masks'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='PadTo', size_divisor=(1, 4, 4)),
    dict(type='RandomCrop', size=224),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='ToTensor', keys=['imgs', 'gt_seg_maps']),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(
        type='Collect',
        keys=['imgs', 'gt_seg_maps'],
        meta_keys=['frame_dir', 'frame_inds', 'original_shape', 'img_shape']
    ),
]

# 对验证/测试保持相同 pipeline，只把 test_mode 设为 True
val_pipeline = [
    dict(
        type='SlidingWindowSampler',
        window_size=window_size,
        stride=stride,
        test_mode=True
    ),
    *train_pipeline[1:]
]

data = dict(
    videos_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=f'{data_root}/train.txt',
        data_prefix=f'{data_root}/frames',
        seg_prefix=f'{data_root}/masks',
        filename_tmpl=filename_tmpl,
        window_size=window_size,
        stride=stride,
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        ann_file=f'{data_root}/val.txt',
        data_prefix=f'{data_root}/frames',
        seg_prefix=f'{data_root}/masks',
        filename_tmpl=filename_tmpl,
        window_size=window_size,
        stride=stride,
        test_mode=True,
        pipeline=val_pipeline
    ),
    test=dict(
        type=dataset_type,
        ann_file=f'{data_root}/test.txt',
        data_prefix=f'{data_root}/frames',
        seg_prefix=f'{data_root}/masks',
        filename_tmpl=filename_tmpl,
        window_size=window_size,
        stride=stride,
        test_mode=True,
        pipeline=val_pipeline
    )
)

# ===================== 优化器和学习策略 =====================
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0.05)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[10, 20])
total_epochs = 30

# ===================== 日志与运行 =====================
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/sliding_window_segmentation'
workflow = [('train', 1)] 