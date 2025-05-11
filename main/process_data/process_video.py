# process_video.py
import os
from pipeline import (RandomCrop, SlidingWindowSampler, Resize, PadTo, ToTensor, Normalize, FormatShape,DecordDecode,DecordInit,
                      LoadMaskFromVideo,Flip)

def preprocess_sample(
                      window_size=32,
                      stride=16,):
    # 1. 构造最初的 results
    results = {
        'mask_video': '/home/chenli/weber/Video-Swin-Transformer/simulated_gas/smoke_only1.mp4',
        'filename': '/home/chenli/weber/Video-Swin-Transformer/simulated_gas/simgasvid1.mp4'
    }
    
   
    # 2. 滑窗采样
    decoder = DecordInit()
    results = decoder(results)
    sampler = SlidingWindowSampler(window_size=window_size,
                                   stride=stride,
                                   test_mode=False)
    results = sampler(results)
    decoder_decode = DecordDecode()
    results=decoder_decode(results)

    
    # 4. 按索引从 mask 视频读出灰度帧并二值化
    mask_loader = LoadMaskFromVideo(
        mask_key='mask_video',
        frame_inds_key='frame_inds')
    results = mask_loader(results)
    # 5. Resize 到 (320,240)
    results = Resize((320,240))(results)
    # 6. 随机裁剪到 (224,224)
    results = RandomCrop(224)(results)
    # 7. 随机水平翻转
    results = Flip(flip_ratio=0.5)(results)
    # 8. Pad 到能被 patch/window 整除
    results = PadTo((2, 4, 4))(results)

    # 9. 转为 Tensor 并归一化
    # 先做归一化
    norm = Normalize()
    results = norm(results)
    # 再转 Tensor
    results = ToTensor(keys=['imgs', 'gt_seg_maps'])(results)

    # 9. 格式化形状
    results = FormatShape(input_format='NCTHW')(results)

    # 此时 results['imgs'] 就是模型要的 Tensor[N, C, T, H, W]
    #     results['gt_seg_maps'] 就是 Tensor[N, T, H, W]
    return results


# 调试入口
if __name__ == "__main__":


    
    # 调用处理函数
    results = preprocess_sample(
        window_size=32,
        stride=16,
    )
    
    # 打印结果的键和值的类型、形状信息
    print("\n=== preprocess_sample 返回结果 ===")
    for key, value in results.items():
        print(f"Key: {key}, Type: {type(value)}")
        if hasattr(value, "shape"):
            print(f"  Shape: {value.shape}")
    
    # 在这里设置断点可查看完整results内容
    print("\n调试完成")




