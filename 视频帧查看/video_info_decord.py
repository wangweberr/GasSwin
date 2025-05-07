#!/usr/bin/env python3
# 视频信息查看工具 - 使用Decord库

import argparse
import os
import sys
from datetime import timedelta

def get_video_info_decord(video_path):
    """使用decord获取视频的基本信息"""
    try:
        import decord
    except ImportError:
        print("错误: 未安装decord库，请使用 pip install decord 安装")
        return None
    
    if not os.path.exists(video_path):
        print(f"错误: 文件 '{video_path}' 不存在")
        return None
    
    try:
        # 使用decord打开视频
        reader = decord.VideoReader(video_path)
        
        # 获取基本信息
        frame_count = len(reader)
        if frame_count > 0:
            # 获取第一帧的形状
            first_frame = reader[0].asnumpy()
            height, width = first_frame.shape[:2]
        else:
            height, width = 0, 0
        
        fps = reader.get_avg_fps()
        
        # 计算时长
        duration_seconds = frame_count / fps if fps > 0 else 0
        duration = str(timedelta(seconds=int(duration_seconds)))
        
        # 返回信息字典
        return {
            "文件名": os.path.basename(video_path),
            "文件大小": f"{os.path.getsize(video_path) / (1024*1024):.2f} MB",
            "总帧数": f"{frame_count:,}",
            "分辨率": f"{width}x{height}",
            "帧率": f"{fps:.2f} fps",
            "时长": duration
        }
    except Exception as e:
        print(f"错误: {str(e)}")
        return None

def verify_frame_count_decord(video_path, sample_frames=10):
    """通过实际读取一部分帧来验证总帧数"""
    try:
        import decord
    except ImportError:
        return "错误: 未安装decord库"
    
    print(f"\n验证帧数中 (抽样 {sample_frames} 帧)...")
    
    try:
        reader = decord.VideoReader(video_path)
        reported_frames = len(reader)
        
        # 抽样位置
        sample_positions = []
        
        # 生成均匀分布的抽样位置
        if reported_frames <= sample_frames:
            sample_positions = list(range(reported_frames))
        else:
            step = reported_frames / sample_frames
            sample_positions = [int(i * step) for i in range(sample_frames)]
            # 确保最后一帧被包括
            if reported_frames-1 not in sample_positions:
                sample_positions[-1] = reported_frames - 1
        
        # 尝试读取帧
        frames = reader.get_batch(sample_positions).asnumpy()
        successfully_read = frames.shape[0]
        
        result = f"抽样 {len(sample_positions)} 帧，成功读取 {successfully_read} 帧"
        if successfully_read < len(sample_positions):
            result += f"\n警告: 有 {len(sample_positions) - successfully_read} 帧无法读取，" \
                     f"报告的总帧数可能不准确"
        
        return result
    except Exception as e:
        return f"验证失败: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="使用Decord显示视频文件的详细信息")
    parser.add_argument("video_path", help="视频文件的路径")
    parser.add_argument("--verify", "-v", action="store_true",
                       help="验证帧数是否准确（通过抽样读取)")
    args = parser.parse_args()
    
    # 获取视频信息
    info = get_video_info_decord(args.video_path)
    if not info:
        sys.exit(1)
    
    # 打印视频信息
    print("\n=== 视频信息 (Decord) ===")
    for key, value in info.items():
        print(f"{key}: {value}")
    
    # 如果需要验证帧数
    if args.verify:
        verification_result = verify_frame_count_decord(args.video_path)
        print(verification_result)

if __name__ == "__main__":
    main() 