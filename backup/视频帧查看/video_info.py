#!/usr/bin/env python3
# 视频信息查看工具 - 使用OpenCV和argparse

import cv2
import argparse
import os
import sys
from datetime import timedelta

def get_video_info(video_path):
    """获取视频的基本信息：总帧数、分辨率、帧率等"""
    # 检查文件是否存在
    if not os.path.exists(video_path):
        print(f"错误: 文件 '{video_path}' 不存在")
        return None
    
    # 尝试打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频 '{video_path}'")
        return None
    
    # 获取视频基本信息
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 获取视频编解码器
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = chr(fourcc_int & 0xFF) + chr((fourcc_int >> 8) & 0xFF) + \
             chr((fourcc_int >> 16) & 0xFF) + chr((fourcc_int >> 24) & 0xFF)
    
    # 计算视频时长
    duration_seconds = frame_count / fps if fps > 0 else 0
    duration = str(timedelta(seconds=int(duration_seconds)))
    
    # 释放视频捕获对象
    cap.release()
    
    # 返回信息字典
    return {
        "文件名": os.path.basename(video_path),
        "文件大小": f"{os.path.getsize(video_path) / (1024*1024):.2f} MB",
        "总帧数": f"{frame_count:,}",
        "分辨率": f"{width}x{height}",
        "帧率": f"{fps:.2f} fps",
        "视频编码": fourcc,
        "时长": duration
    }

def verify_frame_count(video_path, sample_frames=10):
    """通过实际读取一部分帧来验证总帧数"""
    print(f"\n验证帧数中 (抽样 {sample_frames} 帧)...")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "无法打开视频进行验证"
    
    reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 尝试读取最后几帧
    successfully_read = 0
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
    
    for pos in sample_positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, _ = cap.read()
        if ret:
            successfully_read += 1
    
    cap.release()
    
    result = f"抽样 {len(sample_positions)} 帧，成功读取 {successfully_read} 帧"
    if successfully_read < len(sample_positions):
        result += f"\n警告: 有 {len(sample_positions) - successfully_read} 帧无法读取，" \
                 f"报告的总帧数可能不准确"
    
    return result

def main():
    parser = argparse.ArgumentParser(description="显示视频文件的详细信息")
    parser.add_argument("video_path", help="视频文件的路径")
    parser.add_argument("--verify", "-v", action="store_true", 
                       help="验证帧数是否准确（通过抽样读取)")
    args = parser.parse_args()
    
    # 获取视频信息
    info = get_video_info(args.video_path)
    if not info:
        sys.exit(1)
    
    # 打印视频信息
    print("\n=== 视频信息 ===")
    for key, value in info.items():
        print(f"{key}: {value}")
    
    # 如果需要验证帧数
    if args.verify:
        verification_result = verify_frame_count(args.video_path)
        print(verification_result)

if __name__ == "__main__":
    main() 