from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """
    对 BCHW 特征的通道维执行 LayerNorm。

    对于每个空间位置 (h, w)，在 C 维进行归一化。
    """

    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.num_channels = num_channels
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"LayerNorm2d expects BCHW input, "
                f"but received shape {tuple(x.shape)}."
            )

        if x.shape[1] != self.num_channels:
            raise ValueError(
                f"Expected {self.num_channels} channels, "
                f"but received {x.shape[1]} channels."
            )

        # BCHW -> BHWC
        x = x.permute(0, 2, 3, 1)

        x = F.layer_norm(
            x,
            normalized_shape=(self.num_channels,),
            weight=self.weight,
            bias=self.bias,
            eps=self.eps,
        )

        # BHWC -> BCHW
        return x.permute(0, 3, 1, 2).contiguous()


class ConvNormAct(nn.Module):
    """
    Conv2d + GroupNorm + activation。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        use_activation: bool = True,
    ) -> None:
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )

        self.norm = nn.GroupNorm(
            num_groups=self._get_num_groups(out_channels),
            num_channels=out_channels,
        )

        self.act = nn.GELU() if use_activation else nn.Identity()

    @staticmethod
    def _get_num_groups(channels: int) -> int:
        """
        返回可以整除 channels 的最大 GroupNorm 分组数，
        上限为 32。
        """
        for groups in (32, 16, 8, 4, 2, 1):
            if channels % groups == 0:
                return groups

        return 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class LayerAdapter(nn.Module):
    """
    对单个 encoder block 的特征进行适配。

    结构：
        LayerNorm2d
        -> 1x1 Conv
        -> 3x3 Depthwise Conv
        -> GELU
        -> 1x1 Conv

    同时保留一个投影残差分支。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        adapter_scale_init: float = 0.1,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm = LayerNorm2d(in_channels)

        self.input_proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
        )

        self.depthwise_conv = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            groups=out_channels,
            bias=False,
        )

        self.depthwise_norm = nn.GroupNorm(
            num_groups=ConvNormAct._get_num_groups(out_channels),
            num_channels=out_channels,
        )

        self.activation = nn.GELU()

        self.output_proj = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
        )

        # 保证残差分支和 Adapter 分支具有相同通道数。
        if in_channels == out_channels:
            self.skip_proj = nn.Identity()
        else:
            self.skip_proj = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False,
            )

        # 控制 Adapter 分支初始影响程度。
        self.adapter_scale = nn.Parameter(
            torch.tensor(float(adapter_scale_init))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip_proj(x)

        y = self.norm(x)
        y = self.input_proj(y)
        y = self.depthwise_conv(y)
        y = self.depthwise_norm(y)
        y = self.activation(y)
        y = self.output_proj(y)

        return residual + self.adapter_scale * y


class GlobalDepthRouter(nn.Module):
    """
    图像级深度路由器。

    对每个 encoder 层输出一个全局标量分数：

        g_i = MLP([z_i, z_anchor, z_i - z_anchor])

    其中 anchor 默认是最深层特征。
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int = 128,
    ) -> None:
        super().__init__()

        self.score_mlp = nn.Sequential(
            nn.Linear(channels * 3, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, 1),
        )

        self._initialize_last_layer()

    def _initialize_last_layer(self) -> None:
        """
        将最后一层初始化为零，使训练开始时各层全局权重接近均匀。
        """
        last_layer = self.score_mlp[-1]

        if isinstance(last_layer, nn.Linear):
            nn.init.zeros_(last_layer.weight)
            nn.init.zeros_(last_layer.bias)

    def forward(
        self,
        features: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features:
                L 个 BCHW 特征，各层通道数和空间尺寸相同。

        Returns:
            global_logits:
                B x L

            global_weights:
                B x L
        """
        if len(features) == 0:
            raise ValueError("features must contain at least one tensor.")

        anchor = features[-1]
        anchor_vector = F.adaptive_avg_pool2d(
            anchor,
            output_size=1,
        ).flatten(1)

        level_logits: List[torch.Tensor] = []

        for feature in features:
            feature_vector = F.adaptive_avg_pool2d(
                feature,
                output_size=1,
            ).flatten(1)

            router_input = torch.cat(
                [
                    feature_vector,
                    anchor_vector,
                    feature_vector - anchor_vector,
                ],
                dim=1,
            )

            # B x 1
            level_logit = self.score_mlp(router_input)
            level_logits.append(level_logit)

        # B x L
        global_logits = torch.cat(level_logits, dim=1)
        global_weights = F.softmax(global_logits, dim=1)
        # print(
        #     "Fusion weights:",
        #     global_weights.cpu().tolist()
        # )
        return global_logits, global_weights


class SpatialDepthRouter(nn.Module):
    """
    空间位置级深度路由器。

    对每个层级生成一个空间分数图：

        S_i = Conv([X_i, X_anchor, |X_i - X_anchor|])

    输出尺寸为 B x L x H x W。
    """

    def __init__(
        self,
        channels: int,
        num_levels: int,
        hidden_channels: int = 128,
    ) -> None:
        super().__init__()

        self.channels = channels
        self.num_levels = num_levels

        self.level_routers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=channels * 3,
                        out_channels=hidden_channels,
                        kernel_size=1,
                        bias=False,
                    ),
                    nn.GroupNorm(
                        num_groups=ConvNormAct._get_num_groups(
                            hidden_channels
                        ),
                        num_channels=hidden_channels,
                    ),
                    nn.GELU(),
                    nn.Conv2d(
                        in_channels=hidden_channels,
                        out_channels=1,
                        kernel_size=1,
                        bias=True,
                    ),
                )
                for _ in range(num_levels)
            ]
        )

        self._initialize_output_layers()

    def _initialize_output_layers(self) -> None:
        """
        将空间路由器的最后一层初始化为零。

        这样训练初始阶段空间 logits 接近零，
        路由主要受全局先验控制。
        """
        for router in self.level_routers:
            output_layer = router[-1]

            if isinstance(output_layer, nn.Conv2d):
                nn.init.zeros_(output_layer.weight)
                nn.init.zeros_(output_layer.bias)

    def forward(
        self,
        features: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            features:
                L 个 B x C x H x W 特征。

        Returns:
            spatial_logits:
                B x L x H x W
        """
        if len(features) != self.num_levels:
            raise ValueError(
                f"Expected {self.num_levels} feature levels, "
                f"but received {len(features)}."
            )

        anchor = features[-1]
        level_logits: List[torch.Tensor] = []

        for level_index, feature in enumerate(features):
            router_input = torch.cat(
                [
                    feature,
                    anchor,
                    torch.abs(feature - anchor),
                ],
                dim=1,
            )

            # B x 1 x H x W
            level_logit = self.level_routers[level_index](
                router_input
            )
            level_logits.append(level_logit)

        # B x L x H x W
        return torch.cat(level_logits, dim=1)


class DynamicDepthFusion(nn.Module):
    """
    基于全局路由和空间路由的多层特征融合模块。

    公式：

        Q_i(x, y)
            = S_i(x, y)
            + lambda_global * log(g_i + eps)

        alpha_i(x, y)
            = softmax_i(Q_i(x, y) / temperature)

        D_r
            = sum_i alpha_i * X_i

        D
            = X_anchor + gamma * Refine(D_r)

    其中最深层 X_anchor 是稳定的主残差路径。
    """

    def __init__(
        self,
        channels: int,
        num_levels: int,
        global_hidden_channels: int = 128,
        spatial_hidden_channels: int = 128,
        temperature: float = 1.0,
        global_prior_scale: float = 1.0,
        fusion_scale_init: float = 0.1,
    ) -> None:
        super().__init__()

        if num_levels < 2:
            raise ValueError(
                "DynamicDepthFusion requires at least two levels."
            )

        if temperature <= 0:
            raise ValueError("temperature must be greater than zero.")

        self.channels = channels
        self.num_levels = num_levels
        self.temperature = float(temperature)
        self.global_prior_scale = float(global_prior_scale)

        self.global_router = GlobalDepthRouter(
            channels=channels,
            hidden_channels=global_hidden_channels,
        )

        self.spatial_router = SpatialDepthRouter(
            channels=channels,
            num_levels=num_levels,
            hidden_channels=spatial_hidden_channels,
        )

        self.refine = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=ConvNormAct._get_num_groups(channels),
                num_channels=channels,
            ),
            nn.GELU(),
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
                bias=False,
            ),
        )

        # 使模块训练初始状态接近最深层特征。
        self.fusion_scale = nn.Parameter(
            torch.tensor(float(fusion_scale_init))
        )

    def set_temperature(self, temperature: float) -> None:
        """
        用于训练过程中的温度退火。
        """
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero.")

        self.temperature = float(temperature)

    def forward(
        self,
        features: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            features:
                L 个经过 Adapter 处理的特征。

                每个特征尺寸必须为：
                    B x C x H x W

                且所有层的 B、C、H、W 必须相同。

        Returns:
            fused_feature:
                B x C x H x W

            routing_info:
                包含全局权重、空间权重以及相关 logits。
        """
        self._validate_features(features)

        global_logits, global_weights = self.global_router(features)
        B, _, H, W = features[0].shape

        # B x L x H x W
        spatial_logits = self.spatial_router(features)

        # 将全局权重作为空间路由先验。
        #
        # B x L
        # -> B x L x 1 x 1
        global_log_prior = torch.log(
            global_weights.clamp_min(1e-6)
        )[:, :, None, None]

        combined_logits = (
            spatial_logits
            + self.global_prior_scale * global_log_prior
        )

        # 在层级维 L 上做 Softmax。
        #
        # B x L x H x W
        ###########################################################################################################
        # spatial_weights = F.softmax(
        #     combined_logits / self.temperature,
        #     dim=1,
        # )
        spatial_weights = (
            global_weights[:, :, None, None]
            .expand(-1, -1, H, W)
        )
        ###########################################################################################################
        # B x L x C x H x W
        stacked_features = torch.stack(
            list(features),
            dim=1,
        )

        # B x L x 1 x H x W
        expanded_weights = spatial_weights.unsqueeze(2)

        # B x C x H x W
        routed_feature = torch.sum(
            stacked_features * expanded_weights,
            dim=1,
        )

        anchor = features[-1]

        fused_feature = (
            anchor
            + self.fusion_scale * self.refine(routed_feature)
        )

        routing_info = {
            "global_logits": global_logits,
            "global_weights": global_weights,
            "spatial_logits": spatial_logits,
            "combined_logits": combined_logits,
            "spatial_weights": spatial_weights,
            "routed_feature": routed_feature,
        }


        return fused_feature, routing_info

    def _validate_features(
        self,
        features: Sequence[torch.Tensor],
    ) -> None:
        if len(features) != self.num_levels:
            raise ValueError(
                f"Expected {self.num_levels} feature levels, "
                f"but received {len(features)}."
            )

        reference_shape = features[0].shape

        if len(reference_shape) != 4:
            raise ValueError(
                f"Expected BCHW tensors, got shape {reference_shape}."
            )

        if reference_shape[1] != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels, "
                f"but received {reference_shape[1]}."
            )

        for level_index, feature in enumerate(features):
            if feature.ndim != 4:
                raise ValueError(
                    f"Feature {level_index} is not a BCHW tensor: "
                    f"{tuple(feature.shape)}."
                )

            if feature.shape != reference_shape:
                raise ValueError(
                    "All adapted features must have identical shapes. "
                    f"Feature 0 has shape {reference_shape}, "
                    f"but feature {level_index} has shape "
                    f"{tuple(feature.shape)}."
                )


class MultiLevelEncoderFusion(nn.Module):
    """
    完整的前半部分融合模块。

    数据流：

        encoder features
            -> independent Layer Adapters
            -> class-specific dynamic depth fusion
            -> fused feature

    clss 中的类别与融合模块对应关系：

        clss == 0 -> depth_fusion[0]
        clss == 1 -> depth_fusion[1]
        clss == 2 -> depth_fusion[2]

    该模块不构建空间金字塔，只输出一个统一融合特征。
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int = 256,
        adapter_scale_init: float = 0.1,
        global_hidden_channels: int = 128,
        spatial_hidden_channels: int = 128,
        temperature: float = 1.0,
        global_prior_scale: float = 1.0,
        fusion_scale_init: float = 0.1,
    ) -> None:
        super().__init__()

        if len(in_channels) < 2:
            raise ValueError(
                "At least two encoder levels are required."
            )

        self.in_channels = list(in_channels)
        self.out_channels = out_channels
        self.num_levels = len(in_channels)

        # 固定支持类别 0、1、2
        self.num_fusion_classes = 3

        # self.adapters = nn.ModuleList(
        #     [
        #         LayerAdapter(
        #             in_channels=level_channels,
        #             out_channels=out_channels,
        #             adapter_scale_init=adapter_scale_init,
        #         )
        #         for level_channels in in_channels
        #     ]
        # )

        # 三个参数完全独立的融合模块
        self.depth_fusion = nn.ModuleList(
            [
                DynamicDepthFusion(
                    channels=out_channels,
                    num_levels=self.num_levels,
                    global_hidden_channels=global_hidden_channels,
                    spatial_hidden_channels=spatial_hidden_channels,
                    temperature=temperature,
                    global_prior_scale=global_prior_scale,
                    fusion_scale_init=fusion_scale_init,
                )
                for _ in range(self.num_fusion_classes)
            ]
        )

    def set_temperature(self, temperature: float) -> None:
        # 同时设置三个融合模块的 temperature
        for fusion_module in self.depth_fusion:
            fusion_module.set_temperature(temperature)

    def forward(
        self,
        encoder_features: Sequence[torch.Tensor],
        clss: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            encoder_features:
                多个 encoder block 输出。

                支持不同通道数，但空间尺寸必须相同：

                    F1: B x C1 x H x W
                    F2: B x C2 x H x W
                    ...
                    FL: B x CL x H x W

                特征顺序应从浅层到深层排列，
                最后一个特征作为语义锚点和主残差路径。

            clss:
                每个样本对应的类别，形状为：

                    B

                类别只能取 0、1、2：

                    0 -> 第一个融合模块
                    1 -> 第二个融合模块
                    2 -> 第三个融合模块

        Returns:
            fused_feature:
                B x out_channels x H x W

            auxiliary:
                路由权重和适配后特征。

                所有带有 batch 维度的路由信息均会恢复到
                encoder_features 原始的 batch 顺序。
        """
        self._validate_raw_features(encoder_features)


        #######################################################################
        # adapted_features = [
        #     adapter(feature)
        #     for adapter, feature in zip(
        #         self.adapters,
        #         encoder_features,
        #     )
        # ]
        adapted_features = encoder_features
        #######################################################################

        grouped_indices = []
        grouped_fused_features = []
        grouped_routing_info = {}

        expected_routing_keys = None

        # 按类别对 batch 内样本进行分组
        for class_index in range(self.num_fusion_classes):
            sample_indices = torch.nonzero(
                clss == class_index,
                as_tuple=False,
            ).flatten()

            # 当前 batch 中可能没有该类别
            if sample_indices.numel() == 0:
                continue

            class_features = [
                feature.index_select(
                    dim=0,
                    index=sample_indices,
                )
                for feature in adapted_features
            ]

            class_fused_feature, class_routing_info = (
                self.depth_fusion[class_index](class_features)
            )

            current_routing_keys = set(class_routing_info.keys())

            if expected_routing_keys is None:
                expected_routing_keys = current_routing_keys
            elif current_routing_keys != expected_routing_keys:
                raise RuntimeError(
                    "The three DynamicDepthFusion modules returned "
                    "different routing_info keys."
                )

            grouped_indices.append(sample_indices)
            grouped_fused_features.append(class_fused_feature)

            for key, value in class_routing_info.items():
                if not isinstance(value, torch.Tensor):
                    raise TypeError(
                        f"routing_info['{key}'] must be a Tensor, "
                        f"but received {type(value).__name__}."
                    )

                if value.ndim == 0:
                    raise ValueError(
                        f"routing_info['{key}'] must contain a "
                        "batch dimension."
                    )

                if value.shape[0] != sample_indices.numel():
                    raise ValueError(
                        f"routing_info['{key}'] has batch size "
                        f"{value.shape[0]}, but the corresponding "
                        f"class group contains "
                        f"{sample_indices.numel()} samples."
                    )

                grouped_routing_info.setdefault(key, []).append(value)

        # 分组后的 batch 顺序，例如可能为：
        # [类别0样本, 类别1样本, 类别2样本]
        grouped_indices = torch.cat(
            grouped_indices,
            dim=0,
        )

        # 恢复 encoder_features 原始 batch 顺序
        restore_indices = torch.argsort(grouped_indices)

        fused_feature = torch.cat(
            grouped_fused_features,
            dim=0,
        ).index_select(
            dim=0,
            index=restore_indices,
        )

        routing_info = {
            key: torch.cat(values, dim=0).index_select(
                dim=0,
                index=restore_indices,
            )
            for key, values in grouped_routing_info.items()
        }

        auxiliary = {
            **routing_info,
            "adapted_features": adapted_features,
        }

        return fused_feature, auxiliary

    def _validate_raw_features(
        self,
        encoder_features: Sequence[torch.Tensor],
    ) -> None:
        if len(encoder_features) != self.num_levels:
            raise ValueError(
                f"Expected {self.num_levels} encoder features, "
                f"but received {len(encoder_features)}."
            )

        reference_batch = encoder_features[0].shape[0]
        reference_spatial_size = encoder_features[0].shape[-2:]

        for level_index, feature in enumerate(encoder_features):
            if feature.ndim != 4:
                raise ValueError(
                    f"Feature {level_index} must be BCHW, "
                    f"but received shape {tuple(feature.shape)}."
                )

            expected_channels = self.in_channels[level_index]

            if feature.shape[1] != expected_channels:
                raise ValueError(
                    f"Feature {level_index} should have "
                    f"{expected_channels} channels, "
                    f"but received {feature.shape[1]}."
                )

            if feature.shape[0] != reference_batch:
                raise ValueError(
                    "All encoder features must have the same "
                    "batch size."
                )

            if feature.shape[-2:] != reference_spatial_size:
                raise ValueError(
                    "This fusion module expects encoder block "
                    "features with identical spatial resolution. "
                    f"Feature 0 has spatial size "
                    f"{reference_spatial_size}, but feature "
                    f"{level_index} has spatial size "
                    f"{feature.shape[-2:]}."
                )
