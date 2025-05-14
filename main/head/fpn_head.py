# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch.nn as nn
import torch.nn.functional as F



class FPNHead(nn.Module):
    """Panoptic Feature Pyramid Networks.

    This head is the implementation of `Semantic FPN
    <https://arxiv.org/abs/1901.02446>`_.

    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
    """

    def __init__(self, feature_strides=[1,2,4,8], in_channels=[96,192,384,768]):
        super(FPNHead, self).__init__()
        self.in_channels=in_channels
        self.in_index=[0,1,2,3]
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        self.channels=256
        self.scale_heads = nn.ModuleList()
        self.align_corners=True
        self.cls_seg = nn.Conv3d(self.channels, 1, kernel_size=1)
        for i in range(len(feature_strides)):
            head_length = max(
                1,
                int(np.log2(feature_strides[i]) - np.log2(feature_strides[0])))
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
                if feature_strides[i] != feature_strides[0]:
                    scale_head.append(
                        lambda x: F.interpolate(
                            x,
                            scale_factor=2,
                            mode='trilinear',  # 对于3D数据用trilinear
                            align_corners=self.align_corners))
            self.scale_heads.append(nn.Sequential(*scale_head))
        

    def _transform_inputs(self, inputs):
        return [inputs[i] for i in self.in_index]
    def forward(self, inputs):

        x = self._transform_inputs(inputs)

        output = self.scale_heads[0](x[0])
        for i in range(1, len(self.feature_strides)):
            # non inplace
            output = output + F.interpolate(
                self.scale_heads[i](x[i]),
                size=output.shape[2:],
                mode='trilinear',
                align_corners=True)

        output = self.cls_seg(output)
        return output
