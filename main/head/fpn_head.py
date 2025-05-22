# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Interpolate(nn.Module):
    def __init__(self,scale_factor,mode='trilinear',align_corners=True):
        super().__init__()
        self.scale_factor=scale_factor
        self.mode=mode
        self.align_corners=align_corners
    def forward(self,x):
        return F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners)
        

class FPNHead(nn.Module):
    """Panoptic Feature Pyramid Networks.

    This head is the implementation of `Semantic FPN
    <https://arxiv.org/abs/1901.02446>`_.

    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
    """

    def __init__(self, feature_strides=[4,8,16,32], in_channels=[96,192,384,768],raw_in_channels=1):
        super(FPNHead, self).__init__()
        self.in_channels=in_channels
        self.raw_in_channels=raw_in_channels
        self.in_index=[0,1,2,3]
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        self.channels=256
        self.scale_heads = nn.ModuleList()
        self.align_corners=True
        self.cls_seg = nn.Conv3d(self.channels, 1, kernel_size=1)

        # 添加一个卷积层，将raw_in_channels转换为channels
        self.p0_conv = nn.Sequential(
            nn.Conv3d(raw_in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, self.channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(self.channels),
            nn.ReLU(inplace=True)
        )
        for i in range(len(feature_strides)):
            head_length =int(np.log2(feature_strides[i])-1)
            scale_head = []
            for k in range(head_length):
                scale_head.append(
                    nn.Sequential(
                        nn.Conv3d(
                            self.in_channels[i] if k == 0 else self.channels,
                            self.channels,
                            kernel_size=3,
                            padding=1,
                            bias=False  # 通常在使用BN时不需要bias
                        ),
                        nn.BatchNorm3d(self.channels),
                        nn.ReLU(inplace=True)
                    )
                )

                scale_head.append(
                    Interpolate(
                        scale_factor=2,
                        mode='trilinear',  # 对于3D数据用trilinear
                        align_corners=self.align_corners))
            self.scale_heads.append(nn.Sequential(*scale_head))
        


    def forward(self, inputs,raw_input):
        x=[self.scale_heads[i](inputs[i]) for i in range(len(self.feature_strides))]
        p0=self.p0_conv(raw_input)
        for i in range(len(x)):
            # 再次上采样对齐原始数据
            x[i]= F.interpolate(
                x[i],
                size=p0.shape[2:],
                mode='trilinear',
                align_corners=True)
        fused=p0
        for feat in x:
            fused= fused+feat
        output = self.cls_seg(fused)
        return output
