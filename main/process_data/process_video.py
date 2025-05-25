# process_video.py
import os
from .pipeline import (RandomCrop, SlidingWindowSampler, Resize, PadTo, ToTensor, Normalize, FormatShape,DecordDecode,DecordInit,
                      LoadMaskFromVideo,Flip)

def preprocess_sample(results):

    
 
    # 2. 滑窗采样
    decoder = DecordInit()
    results = decoder(results)
    sampler = SlidingWindowSampler(results,test_mode=False)
    results = sampler(results)
    decoder_decode = DecordDecode()
    results=decoder_decode(results)

    
    # 4. 按索引从 mask 视频读出灰度帧并二值化
    mask_loader = LoadMaskFromVideo(
        mask_video='mask_video',
        frame_inds='frame_inds')
    results = mask_loader(results)
    # 5. Resize 到 (320,240)
    results = Resize((320,240))(results)
    # 6. 随机裁剪到 (224,224)
    #results = RandomCrop((224,224))(results)
    # 7. 随机水平翻转
    #results = Flip(flip_ratio=0.5)(results)
    # 8. Pad 到能被 patch/window 整除
    results = PadTo((2, 4, 4))(results)

    # 9. 转为 Tensor 并归一化
    # 先做归一化
    norm = Normalize()
    results = norm(results)
    # 再转 Tensor
    results = ToTensor(keys=['imgs', 'gt_seg_maps'])(results)#THW

    # 9. 格式化形状
    results = FormatShape(input_format='THW')(results)

    # 此时 results['imgs'] 就是模型要的 Tensor[N, C, T, H, W]
    #     results['gt_seg_maps'] 就是 Tensor[N, C, T, H, W]
    return results


# 调试入口
if __name__ == "__main__":


    results = {
        'mask_video': '/home/chenli/weber/Video-Swin-Transformer/simulated_gas/labels/pop',
        'filename': '/home/chenli/weber/Video-Swin-Transformer/simulated_gas/gas/pop',
        'model_path':'/home/chenli/weber/Video-Swin-Transformer/main/checkpoints/models',
        'pretrained':'/home/chenli/weber/Video-Swin-Transformer/pretrained/swin_tiny_patch244_window877_kinetics400_1k.pth',
        # 视频处理相关字段
        'video_readers': [],     # 视频读取器列表
        'total_frames': [],      # 各视频总帧数
        'frame_inds': [],        # 采样帧索引列表
        # 图像数据存储
        'imgs': [],              # 原始视频帧列表
        'original_shape': [],    # 原始视频尺寸
        # 掩码处理相关
        'mask_paths': [],        # 掩码视频路径列表
        'gt_seg_maps': [],       # 二值化掩码数据
        # 预处理状态
        'img_shape': [],         # 当前处理尺寸
        'flip': [],              # 翻转状态
        # 数据加载参数
        'clip_len': 32,         # 采样窗口大小
        'frame_interval': 1,    # 帧间隔
        'batch_size': 16,
        'test_size': 0.2,
        'val_size': 0.1,
        'num_workers': 4,
        'video_indices':[],
        'mask_location':[],
        'video_paths':[],
        'window_size':32,
        'stride':16,
        'epochs':20,
        'patience':10,
        'pretrained2d':False
    }
    # 调用处理函数
    results = preprocess_sample(results)
    
    # 打印结果的键和值的类型、形状信息
    print("\n=== preprocess_sample 返回结果 ===")
    print(results['imgs'][0].shape)
    
    # 在这里设置断点可查看完整results内容
    print("\n调试完成")




