# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
#融合模块， 3是std+global 2是std
"""Necks are the interface between a vision backbone and the rest of the detection model"""

from copy import deepcopy
from typing import List, Optional, Tuple

import torch

import torch.nn as nn
# from .mlfusion import MultiLevelEncoderFusion
#global newlr
# from .mlfusion2 import MultiLevelEncoderFusion
#unknown
from .mlfusion3 import MultiLevelEncoderFusion
#global+std
class Sam3DualViTDetNeck(nn.Module):
    def __init__(
        self,
        trunk: nn.Module,
        position_encoding: nn.Module,
        d_model: int,
        scale_factors=(4.0, 2.0, 1.0, 0.5),
        add_sam2_neck: bool = False,
    ):
        """
        SimpleFPN neck a la ViTDet
        (From detectron2, very lightly adapted)
        It supports a "dual neck" setting, where we have two identical necks (for SAM3 and SAM2), with different weights

        :param trunk: the backbone
        :param position_encoding: the positional encoding to use
        :param d_model: the dimension of the model
        """
        super().__init__()
        self.trunk = trunk
        self.position_encoding = position_encoding
        self.convs = nn.ModuleList()

        self.scale_factors = scale_factors
        use_bias = True
        dim: int = self.trunk.channel_list[-1]

        for _, scale in enumerate(scale_factors):
            current = nn.Sequential()

            if scale == 4.0:
                current.add_module(
                    "dconv_2x2_0",
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                )
                current.add_module(
                    "gelu",
                    nn.GELU(),
                )
                current.add_module(
                    "dconv_2x2_1",
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                )
                out_dim = dim // 4
            elif scale == 2.0:
                current.add_module(
                    "dconv_2x2",
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                )
                out_dim = dim // 2
            elif scale == 1.0:
                out_dim = dim
            elif scale == 0.5:
                current.add_module(
                    "maxpool_2x2",
                    nn.MaxPool2d(kernel_size=2, stride=2),
                )
                out_dim = dim
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

            current.add_module(
                "conv_1x1",
                nn.Conv2d(
                    in_channels=out_dim,
                    out_channels=d_model,
                    kernel_size=1,
                    bias=use_bias,
                ),
            )
            current.add_module(
                "conv_3x3",
                nn.Conv2d(
                    in_channels=d_model,
                    out_channels=d_model,
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                ),
            )
            self.convs.append(current)

        self.sam2_convs = None
        if add_sam2_neck:
            # Assumes sam2 neck is just a clone of the original neck
            self.sam2_convs = deepcopy(self.convs)
        #########################################################################
        self.fusion_module = MultiLevelEncoderFusion(
            in_channels=[1024, 1024, 1024, 1024],
            out_channels=1024,
            adapter_scale_init=0.1,
            global_hidden_channels=128,
            spatial_hidden_channels=128,
            temperature=1.5,
            global_prior_scale=1.0,
            fusion_scale_init=0.1,
        )

    def forward(
        self, tensor_list: List[torch.Tensor], clss
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        Optional[List[torch.Tensor]],
        Optional[List[torch.Tensor]],
    ]:
        xs = self.trunk(tensor_list, clss)
        sam3_out, sam3_pos = [], []
        sam2_out, sam2_pos = None, None
        if self.sam2_convs is not None:
            sam2_out, sam2_pos = [], []

        # x = xs[-1]  # simpleFPN
# 
        x, auxiliary = self.fusion_module(xs, clss)

        for i in range(len(self.convs)):
            sam3_x_out = self.convs[i](x)
            sam3_pos_out = self.position_encoding(sam3_x_out).to(sam3_x_out.dtype)
            sam3_out.append(sam3_x_out)
            sam3_pos.append(sam3_pos_out)

            if self.sam2_convs is not None:
                sam2_x_out = self.sam2_convs[i](x)
                sam2_pos_out = self.position_encoding(sam2_x_out).to(sam2_x_out.dtype)
                sam2_out.append(sam2_x_out)
                sam2_pos.append(sam2_pos_out)
        return sam3_out, sam3_pos, sam2_out, sam2_pos



# def test_fusion_module() -> None:
#     batch_size = 2
#     height = 64
#     width = 64

#     encoder_features = [
#         torch.randn(batch_size, 1024, height, width),
#         torch.randn(batch_size, 1024, height, width),
#         torch.randn(batch_size, 1024, height, width),
#         torch.randn(batch_size, 1024, height, width),
#     ]

#     fusion_module = MultiLevelEncoderFusion(
#         in_channels=[768, 768, 1024, 1024],
#         out_channels=256,
#         adapter_scale_init=0.1,
#         global_hidden_channels=128,
#         spatial_hidden_channels=128,
#         temperature=1.5,
#         global_prior_scale=1.0,
#         fusion_scale_init=0.1,
#     )

#     fused_feature, auxiliary = fusion_module(
#         encoder_features
#     )

#     print("Fused feature:", fused_feature.shape)
#     print(
#         "Global weights:",
#         auxiliary["global_weights"].shape,
#     )
#     print(
#         "Spatial weights:",
#         auxiliary["spatial_weights"].shape,
#     )

    # # 检查每个像素位置的层级权重和是否为 1。
    # weight_sum = auxiliary["spatial_weights"].sum(dim=1)

    # print(
    #     "Spatial weight sum range:",
    #     weight_sum.min().item(),
    #     weight_sum.max().item(),
    # )
