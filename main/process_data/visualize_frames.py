# visualize_frames.py
import os
import cv2
import numpy as np
import argparse
from .pipeline import (SlidingWindowSampler, DecordInit, DecordDecode, 
                      LoadMaskFromVideo)
#python -m main.process_data.visualize_frames --segment 3
# 创建保存输出的目录
output_dir = "mask_debug"
os.makedirs(output_dir, exist_ok=True)

def save_frame_as_png(frame, filename, auto_scale=True):
    """保存单帧为PNG图像
    
    Args:
        frame: 输入图像帧
        filename: 保存文件名
        auto_scale: 是否自动将[0,1]范围缩放到[0,255]
    """
    # 确保值在0-255范围并转换为uint8类型
    if frame.dtype == np.float32 or frame.dtype == np.float64:
        if auto_scale and frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        else:
            frame = (frame.clip(0, 255)).astype(np.uint8)
    elif frame.max() > 255:
        frame = (np.clip(frame, 0, 255)).astype(np.uint8)
    
    # 如果不是uint8类型，强制转换
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    
    # 灰度图转RGB (如果是单通道)
    if len(frame.shape) == 2:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        frame_rgb = frame
        
    # 保存图像
    cv2.imwrite(filename, frame_rgb)
    print(f"已保存: {filename}")

# 主处理流程
def visualize_mask_processing(segment_idx=0):
    # 准备初始数据字典
    results = {
            'mask_video': '/home/chenli/weber/Video-Swin-Transformer/simulated_gas/labels/pop',
            'filename': '/home/chenli/weber/Video-Swin-Transformer/simulated_gas/gas/pop',
            # 视频处理相关字段
            'video_readers': [],     # 视频读取器列表
            'total_frames': [],      # 各视频总帧数
            'frame_inds': [],        # 采样帧索引列表
            'frame_size': 18,        # 采样窗口大小
            'frame_stride': 9,      # 帧间隔
            # 图像数据存储
            'imgs': [],              # 原始视频帧列表
            'original_shape': [],    # 原始视频尺寸
            # 掩码处理相关
            'mask_paths': [],        # 掩码视频路径列表
            'gt_seg_maps': [],       # 软gt掩码数据
            # 预处理状态
            'img_shape': [],         # 当前处理尺寸
            'flip': [],              # 翻转状态
            # 数据加载参数
            'clip_len': 32,         # 采样窗口大小
            'frame_interval': 1,    # 帧间隔
            'batch_size': 1,
            'test_size': 0.2,
            'val_size': 0.1,
            'num_workers': 4,
            'video_indices':[],
            'mask_location':[],
            'video_paths':[],
            'window_size':32,
            'epochs':60,
            'patience':20,
            'pretrained2d':False
        }
    
    # 1. 初始化视频解码器
    
    decoder_init = DecordInit()
    results = decoder_init(results)
    print(f"视频总帧数: {results['total_frames']}")
    
    # 2. 滑窗采样
    sampler = SlidingWindowSampler(results, test_mode=False)
    results = sampler(results)
    print(f"采样的帧索引: {results['frame_inds']}")
    
    # 3. 解码视频帧
    decoder = DecordDecode()
    results = decoder(results)
    print(f"解码后视频片段数量: {len(results['imgs'])}")
    
    # 4. 加载软gt掩码处理
    mask_loader = LoadMaskFromVideo(mask_video='mask_video', frame_inds='frame_inds')
    results = mask_loader(results)
    print(f"掩码片段数量: {len(results['gt_seg_maps'])}")
    
    # 检查指定片段索引是否有效
    if segment_idx >= len(results['gt_seg_maps']):
        print(f"错误: 指定的片段索引 {segment_idx} 超出了范围。总片段数: {len(results['gt_seg_maps'])}")
        print(f"请指定0至{len(results['gt_seg_maps'])-1}之间的索引值")
        return
    
    # 创建特定片段的输出目录
    segment_dir = f"{output_dir}/segment_{segment_idx}"
    os.makedirs(segment_dir, exist_ok=True)
    
    # 5. 保存软gt掩码序列 (使用指定的滑动窗口片段)
    if len(results['gt_seg_maps']) > 0:
        masks = results['gt_seg_maps'][segment_idx]  # 取指定片段
        
        # 打印掩码数据类型信息，帮助调试
        print(f"掩码数据类型: {masks[0].dtype}")
        print(f"掩码值范围: [{np.min(masks[0])}, {np.max(masks[0])}]")
        print(f"掩码中非零值数量: {np.count_nonzero(masks[0])}")
        
        # 保存该片段的所有帧
        for i, mask in enumerate(masks):
            try:
                # mask是软gt掩码（[0,1]范围，经过LoadMaskFromVideo处理）
                # 恢复原掩码（[0,255]范围，未处理过的原始灰度值）
                original_mask = (mask * 255).astype(np.uint8)
                
                # 保存软gt掩码（乘以128显示为较暗图像，保持原始相对灰度关系）
                soft_gt_display = (mask * 128).astype(np.uint8)
                save_frame_as_png(soft_gt_display, f"{segment_dir}/soft_gt_mask_{i:03d}.png", auto_scale=False)
                
                # 保存原掩码（[0,255]范围，恢复的原始灰度）
                save_frame_as_png(original_mask, f"{segment_dir}/original_mask_{i:03d}.png")
                
                # 同时保存对应的原始帧
                if len(results['imgs']) > 0 and i < len(results['imgs'][segment_idx]):
                    orig_frame = results['imgs'][segment_idx][i]
                    save_frame_as_png(orig_frame, f"{segment_dir}/orig_frame_{i:03d}.png")
                    
                    # 创建叠加效果图（使用原掩码）
                    if mask.shape[:2] == orig_frame.shape[:2]:
                        # 确保原始帧是RGB格式
                        if len(orig_frame.shape) == 2:
                            orig_frame_uint8 = orig_frame.astype(np.uint8)
                            frame_rgb = cv2.cvtColor(orig_frame_uint8, cv2.COLOR_GRAY2BGR)
                        else:
                            frame_rgb = orig_frame.astype(np.uint8)
                        
                        # 创建红色半透明掩码（使用原掩码的灰度信息）
                        overlay = np.zeros_like(frame_rgb)
                        overlay[:,:,2] = original_mask  # 红色通道，使用原掩码
                        
                        # 叠加
                        alpha = 0.7  # 稍微加大透明度使掩码更明显
                        blend = cv2.addWeighted(frame_rgb, 1, overlay, alpha, 0)
                        save_frame_as_png(blend, f"{segment_dir}/overlay_{i:03d}.png")
                        
            except Exception as e:
                print(f"处理第{i}帧时出错: {e}")
                print(f"掩码形状: {mask.shape}, 数据类型: {mask.dtype}")
                if len(results['imgs']) > 0 and i < len(results['imgs'][segment_idx]):
                    orig_frame = results['imgs'][segment_idx][i]
                    print(f"原始帧形状: {orig_frame.shape}, 数据类型: {orig_frame.dtype}")
    
    print(f"\n处理完成! 所有图像已保存到 {segment_dir} 目录")
    print(f"保存了第 {segment_idx} 个片段的 {len(masks)} 帧图像")
    print(f"- 原始帧: orig_frame_XXX.png")
    print(f"- 软gt掩码: soft_gt_mask_XXX.png ([0,1]范围)")
    print(f"- 原掩码: original_mask_XXX.png ([0,255]范围，恢复的原始灰度)")
    print(f"- 叠加效果: overlay_XXX.png (红色原掩码叠加)")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='可视化视频掩码处理效果')
    parser.add_argument('--segment', type=int, default=0, 
                        help='要处理的片段索引 (从0开始，默认为0)')
    args = parser.parse_args()
    
    # 处理指定片段
    visualize_mask_processing(args.segment)

if __name__ == "__main__":
    main() 