import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from typing import Union, Type, List, Tuple, Optional, Dict

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.building_blocks.helper import (
    maybe_convert_scalar_to_list,
    convert_conv_op_to_dim,
    get_matching_instancenorm,
    convert_dim_to_conv_op,
)
from dynamic_network_architectures.building_blocks.residual import BasicBlockD
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.utilities.network_initialization import InitWeights_He


# ---------- 基础模块 ----------
class UpsampleLayer(nn.Module):
    def __init__(
        self,
        conv_op: Type[_ConvNd],
        input_channels: int,
        output_channels: int,
        pool_op_kernel_size,
        mode: str = 'nearest',
        bias: bool = True,
    ):
        super().__init__()
        self.conv = conv_op(input_channels, output_channels, kernel_size=1, bias=bias)
        if isinstance(pool_op_kernel_size, (list, tuple)):
            self.scale_factor = tuple(float(x) for x in pool_op_kernel_size)
        else:
            self.scale_factor = (float(pool_op_kernel_size),)
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return self.conv(x)


class BasicResBlock(nn.Module):
    def __init__(
        self,
        conv_op: Type[_ConvNd],
        input_channels: int,
        output_channels: int,
        norm_op: Union[None, Type[nn.Module]],
        norm_op_kwargs: dict,
        kernel_size=3,
        padding=1,
        stride=1,
        use_1x1conv=False,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={'inplace': True},
        conv_bias: bool = False,
    ):
        super().__init__()
        self.conv1 = conv_op(input_channels, output_channels, kernel_size, stride=stride, padding=padding, bias=conv_bias)
        self.norm1 = norm_op(output_channels, **norm_op_kwargs) if norm_op is not None else nn.Identity()
        self.act1 = nonlin(**nonlin_kwargs) if nonlin is not None else nn.Identity()

        self.conv2 = conv_op(output_channels, output_channels, kernel_size, padding=padding, bias=conv_bias)
        self.norm2 = norm_op(output_channels, **norm_op_kwargs) if norm_op is not None else nn.Identity()
        self.act2 = nonlin(**nonlin_kwargs) if nonlin is not None else nn.Identity()

        self.conv3 = conv_op(input_channels, output_channels, kernel_size=1, stride=stride, bias=True) if use_1x1conv else None

    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(self.norm1(y))
        y = self.norm2(self.conv2(y))
        if self.conv3 is not None:
            x = self.conv3(x)
        y = y + x
        return self.act2(y)


class FusionBlock(nn.Module):
    """多模态通道拼接后的 3x3(×3) 卷积融合，降回单模态通道数。"""
    def __init__(
        self,
        conv_op: Type[_ConvNd],
        input_channels: int,
        output_channels: int,
        norm_op: Union[None, Type[nn.Module]],
        norm_op_kwargs: dict,
        nonlin: Union[None, Type[nn.Module]],
        nonlin_kwargs: dict,
        conv_bias: bool = False,
    ):
        super().__init__()
        dim = convert_conv_op_to_dim(conv_op)
        k = (3,) * dim
        p = (1,) * dim
        self.block = nn.Sequential(
            conv_op(input_channels, output_channels, kernel_size=k, padding=p, bias=conv_bias),
            norm_op(output_channels, **(norm_op_kwargs or {})) if norm_op is not None else nn.Identity(),
            nonlin(**(nonlin_kwargs or {})) if nonlin is not None else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)


# ---------- 编码器 ----------
class UNetResEncoder(nn.Module):
    """return_skips=True：返回浅->深的特征列表"""
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
        n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        nonlin: Union[None, Type[nn.Module]] = None,
        nonlin_kwargs: dict = None,
        return_skips: bool = False,
        stem_channels: int = None,
    ):
        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages

        assert len(kernel_sizes) == n_stages
        assert len(n_blocks_per_stage) == n_stages
        assert len(features_per_stage) == n_stages
        assert len(strides) == n_stages

        dim = convert_conv_op_to_dim(conv_op)
        kernel_sizes = [ks if isinstance(ks, (list, tuple)) else (ks,) * dim for ks in kernel_sizes]
        self.kernel_sizes = kernel_sizes
        self.conv_pad_sizes = [tuple(k // 2 for k in ks) for ks in kernel_sizes]

        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs or {}
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs or {}
        self.conv_bias = conv_bias
        self.return_skips = return_skips

        stem_channels = stem_channels or features_per_stage[0]

        stem_blocks = [
            BasicResBlock(
                conv_op=conv_op,
                input_channels=input_channels,
                output_channels=stem_channels,
                norm_op=self.norm_op,
                norm_op_kwargs=self.norm_op_kwargs,
                kernel_size=self.kernel_sizes[0],
                padding=self.conv_pad_sizes[0],
                stride=1,
                nonlin=self.nonlin,
                nonlin_kwargs=self.nonlin_kwargs,
                use_1x1conv=True,
                conv_bias=self.conv_bias,
            )
        ]
        stem_blocks += [
            BasicBlockD(
                conv_op=conv_op,
                input_channels=stem_channels,
                output_channels=stem_channels,
                kernel_size=self.kernel_sizes[0],
                stride=1,
                conv_bias=self.conv_bias,
                norm_op=self.norm_op,
                norm_op_kwargs=self.norm_op_kwargs,
                nonlin=self.nonlin,
                nonlin_kwargs=self.nonlin_kwargs,
            ) for _ in range(n_blocks_per_stage[0] - 1)
        ]
        self.stem = nn.Sequential(*stem_blocks)

        stages = []
        in_ch = stem_channels
        for s in range(n_stages):
            stage = [
                BasicResBlock(
                    conv_op=conv_op,
                    input_channels=in_ch,
                    output_channels=features_per_stage[s],
                    norm_op=self.norm_op,
                    norm_op_kwargs=self.norm_op_kwargs,
                    kernel_size=self.kernel_sizes[s],
                    padding=self.conv_pad_sizes[s],
                    stride=maybe_convert_scalar_to_list(conv_op, strides[s]),
                    use_1x1conv=True,
                    nonlin=self.nonlin,
                    nonlin_kwargs=self.nonlin_kwargs,
                    conv_bias=self.conv_bias,
                )
            ]
            stage += [
                BasicBlockD(
                    conv_op=conv_op,
                    input_channels=features_per_stage[s],
                    output_channels=features_per_stage[s],
                    kernel_size=self.kernel_sizes[s],
                    stride=1,
                    conv_bias=self.conv_bias,
                    norm_op=self.norm_op,
                    norm_op_kwargs=self.norm_op_kwargs,
                    nonlin=self.nonlin,
                    nonlin_kwargs=self.nonlin_kwargs,
                ) for _ in range(n_blocks_per_stage[s] - 1)
            ]
            stages.append(nn.Sequential(*stage))
            in_ch = features_per_stage[s]

        self.stages = nn.Sequential(*stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]

    def forward(self, x):
        x = self.stem(x)
        ret = []
        for s in self.stages:
            x = s(x)
            ret.append(x)
        return ret if self.return_skips else ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        return np.int64(0)


# ---------- 解码器 ----------
class UNetResDecoder(nn.Module):
    def __init__(
        self,
        encoder: UNetResEncoder,
        num_classes: int,
        n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
        deep_supervision: bool,
        # --- text modulation (optional) ---
        text_embed_dim: Optional[int] = None,
        modulation: str = 'none',  # 'none' | 'film' | 'gate'
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        self.text_embed_dim = text_embed_dim
        self.modulation = modulation

        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1

        stages = []
        upsample_layers = []
        seg_layers = []

        # create per-stage modules
        self.modulators = nn.ModuleList([])
        for s in range(1, n_stages_encoder):
            ch_below = encoder.output_channels[-s]
            ch_skip = encoder.output_channels[-(s + 1)]
            stride_for_upsampling = encoder.strides[-s]

            upsample_layers.append(UpsampleLayer(
                conv_op=encoder.conv_op,
                input_channels=ch_below,
                output_channels=ch_skip,
                pool_op_kernel_size=stride_for_upsampling,
                mode='nearest',
                bias=encoder.conv_bias,
            ))

            stage_blocks = [
                BasicResBlock(
                    conv_op=encoder.conv_op,
                    input_channels=2 * ch_skip,
                    output_channels=ch_skip,
                    norm_op=encoder.norm_op,
                    norm_op_kwargs=encoder.norm_op_kwargs,
                    kernel_size=encoder.kernel_sizes[-(s + 1)],
                    padding=encoder.conv_pad_sizes[-(s + 1)],
                    stride=1,
                    use_1x1conv=True,
                    nonlin=encoder.nonlin,
                    nonlin_kwargs=encoder.nonlin_kwargs,
                    conv_bias=encoder.conv_bias,
                )
            ]
            stage_blocks += [
                BasicBlockD(
                    conv_op=encoder.conv_op,
                    input_channels=ch_skip,
                    output_channels=ch_skip,
                    kernel_size=encoder.kernel_sizes[-(s + 1)],
                    stride=1,
                    conv_bias=encoder.conv_bias,
                    norm_op=encoder.norm_op,
                    norm_op_kwargs=encoder.norm_op_kwargs,
                    nonlin=encoder.nonlin,
                    nonlin_kwargs=encoder.nonlin_kwargs,
                ) for _ in range(n_conv_per_stage[s - 1] - 1)
            ]
            stages.append(nn.Sequential(*stage_blocks))

            seg_layers.append(encoder.conv_op(ch_skip, num_classes, 1, 1, 0, bias=True))

            # Optional: text-based FiLM/gating after the conv stage
            if self.text_embed_dim is not None and self.modulation in ('film', 'gate'):
                hidden = max(32, ch_skip // 2)
                if self.modulation == 'film':
                    self.modulators.append(
                        nn.Sequential(
                            nn.Linear(self.text_embed_dim, hidden),
                            nn.ReLU(inplace=True),
                            nn.Linear(hidden, 2 * ch_skip),  # gamma,beta
                        )
                    )
                else:  # gate
                    self.modulators.append(
                        nn.Sequential(
                            nn.Linear(self.text_embed_dim, hidden),
                            nn.ReLU(inplace=True),
                            nn.Linear(hidden, ch_skip),  # alpha
                        )
                    )
            else:
                self.modulators.append(nn.Identity())

        self.stages = nn.ModuleList(stages)
        self.upsample_layers = nn.ModuleList(upsample_layers)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips: List[torch.Tensor], text: Optional[torch.Tensor] = None):
        lres_input = skips[-1]
        seg_outputs = []

        for s in range(len(self.stages)):
            x = self.upsample_layers[s](lres_input)
            x = torch.cat((x, skips[-(s + 2)]), 1)
            x = self.stages[s](x)

            # apply optional text modulation
            if not isinstance(self.modulators[s], nn.Identity):
                assert text is not None, 'Text embedding must be provided when modulation is enabled.'
                # text: [B, D]
                b, c = x.shape[0], x.shape[1]
                params = self.modulators[s](text)  # [B, P]
                if self.modulation == 'film':
                    gamma, beta = params.chunk(2, dim=-1)
                    gamma = gamma.view(b, c, 1, *([1] * (x.ndim - 3)))
                    beta = beta.view(b, c, 1, *([1] * (x.ndim - 3)))
                    x = gamma * x + beta
                else:
                    alpha = torch.sigmoid(params).view(b, c, 1, *([1] * (x.ndim - 3)))
                    x = alpha * x

            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))

            lres_input = x

        seg_outputs = seg_outputs[::-1]
        return seg_outputs if self.deep_supervision else seg_outputs[0]

    def compute_conv_feature_map_size(self, input_size):
        return np.int64(0)


# ---------- 多编码器主体 ----------
class MultiEncoderUNet(nn.Module):
    """
    多编码器（每模态一个）+ 逐层融合 + 共享解码器。
    输入: [B, M, ...]，M=模态数（由 num_input_channels 传入）。
    """
    def __init__(
        self,
        input_channels: int,   # = 模态数
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
        num_classes: int,
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,   # 占位
        dropout_op_kwargs: dict = None,                     # 占位
        nonlin: Union[None, Type[nn.Module]] = None,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = False,
        stem_channels: int = None,
        # ---- text & alignment (all optional; defaults keep old behavior) ----
        text_embed_dim: Optional[int] = None,
        use_alignment_head: bool = False,
        return_heatmap: bool = False,
        heatmap_temperature: float = 1.0,
        modulation: str = 'none',  # 'none' | 'film' | 'gate'
    ):
        super().__init__()
        num_modalities = input_channels
        self.text_embed_dim = text_embed_dim
        self.modulation = modulation
        self.use_alignment_head = use_alignment_head
        self.return_heatmap = return_heatmap
        self.heatmap_temperature = heatmap_temperature

        n_blocks_per_stage = n_conv_per_stage
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)

        self.encoders = nn.ModuleList([
            UNetResEncoder(
                input_channels=1,
                n_stages=n_stages,
                features_per_stage=features_per_stage,
                conv_op=conv_op,
                kernel_sizes=kernel_sizes,
                strides=strides,
                n_blocks_per_stage=n_blocks_per_stage,
                conv_bias=conv_bias,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs or {},
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs or {},
                return_skips=True,
                stem_channels=stem_channels,
            ) for _ in range(num_modalities)
        ])

        bottleneck_features_in = self.encoders[0].output_channels[-1] * num_modalities
        bottleneck_features_out = self.encoders[0].output_channels[-1]
        self.bottleneck_fusion = FusionBlock(
            conv_op, bottleneck_features_in, bottleneck_features_out,
            norm_op, norm_op_kwargs or {}, nonlin, nonlin_kwargs or {}, conv_bias=conv_bias
        )

        self.skip_fusions = nn.ModuleList()
        for i in range(n_stages - 1):
            skip_in = self.encoders[0].output_channels[i] * num_modalities
            skip_out = self.encoders[0].output_channels[i]
            self.skip_fusions.append(
                FusionBlock(
                    conv_op, skip_in, skip_out,
                    norm_op, norm_op_kwargs or {}, nonlin, nonlin_kwargs or {}, conv_bias=conv_bias
                )
            )

        self.decoder = UNetResDecoder(
            self.encoders[0], num_classes, n_conv_per_stage_decoder, deep_supervision,
            text_embed_dim=self.text_embed_dim, modulation=self.modulation
        )

        # Alignment head (project features to text space and compute cosine sim)
        if self.use_alignment_head and self.text_embed_dim is not None:
            # choose bottleneck features for alignment
            bottleneck_channels = self.encoders[0].output_channels[-1]
            dim = convert_conv_op_to_dim(conv_op)
            k1 = (1,) * dim
            self.align_conv = conv_op(bottleneck_channels, self.text_embed_dim, kernel_size=k1, bias=True)
            self.text_proj = nn.Linear(self.text_embed_dim, self.text_embed_dim, bias=False)
        else:
            self.align_conv = None
            self.text_proj = None

    def forward(self, x, text: Optional[torch.Tensor] = None,
                return_extra: bool = False) -> Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]:
        modalities = list(x.split(1, dim=1))  # M × [B,1,...]
        all_skips = [enc(m) for enc, m in zip(self.encoders, modalities)]

        fused_skips: List[torch.Tensor] = []
        num_skip_levels = len(all_skips[0])
        for i in range(num_skip_levels):
            cat_i = torch.cat([sk[i] for sk in all_skips], dim=1)
            if i == num_skip_levels - 1:
                fused = self.bottleneck_fusion(cat_i)
            else:
                fused = self.skip_fusions[i](cat_i)
            fused_skips.append(fused)

        seg = self.decoder(fused_skips, text=text)

        if self.use_alignment_head and self.align_conv is not None and text is not None:
            # alignment on bottleneck feature map
            fb = fused_skips[-1]
            feat = self.align_conv(fb)                      # [B, D, *spatial]
            # l2 normalize along channel
            feat = F.normalize(feat, dim=1, eps=1e-6)
            t = self.text_proj(text)
            t = F.normalize(t, dim=-1, eps=1e-6)            # [B, D]
            # reshape text to [B, D, 1, 1, ...]
            view_shape = [t.shape[0], t.shape[1]] + [1] * (feat.ndim - 2)
            t = t.view(*view_shape)
            sim = (feat * t).sum(dim=1, keepdim=True)       # cosine sim
            if self.return_heatmap:
                # upsample to highest resolution of decoder output
                if isinstance(seg, list):
                    target_spatial = seg[0].shape[2:]
                else:
                    target_spatial = seg.shape[2:]
                sim_up = F.interpolate(sim, size=target_spatial, mode='trilinear' if len(target_spatial) == 3 else 'bilinear', align_corners=False)
                sim_sig = torch.sigmoid(sim_up / max(1e-6, self.heatmap_temperature))
            else:
                sim_sig = torch.sigmoid(sim / max(1e-6, self.heatmap_temperature))

            if return_extra:
                return {'seg': seg, 'sim': sim_sig}
        return seg

    def compute_conv_feature_map_size(self, input_size):
        return np.int64(0)


# ---------- from_plans 入口 ----------
def _features_per_stage_from_cfg(cfg: ConfigurationManager) -> List[int]:
    return [min(cfg.UNet_base_num_features * 2 ** i, cfg.unet_max_num_features)
            for i in range(len(cfg.conv_kernel_sizes))]

def get_multi_encoder_unet_3d_from_plans(
    plans_manager: PlansManager,
    dataset_json: dict,
    configuration_manager: ConfigurationManager,
    num_input_channels: int,
    deep_supervision: bool = True
):
    """
    兼容多版本 ConfigurationManager：
      - 卷积核尺寸：优先 cm.conv_kernel_sizes / cm.kernel_sizes / cm.get_conv_kernel_sizes()
                   若都不存在则回退为每个 stage 使用 [3,3,3]
      - 下采样步长（pool_op_kernel_sizes）：优先 cm.pool_op_kernel_sizes，
                   若没有则从 plans_manager 或 cm.plans_manager 兜底，
                   仍失败则用 [[2,2,2], ...]（除最后一层为 [1,1,1]）
    """
    cm = configuration_manager

    # ---------- helpers ----------
    def _resolve_pool_kernels():
        # 1) 直接字段（新老版本均常见）
        if hasattr(cm, "pool_op_kernel_sizes"):
            pk = cm.pool_op_kernel_sizes
            if pk is not None and len(pk) > 0:
                return pk

        # 2) 从 plans_manager/配置名兜底
        cfg_name = getattr(cm, "configuration_name", None)
        pm = getattr(cm, "plans_manager", None)
        candidates = []
        for src in (pm, plans_manager):
            if src is None:
                continue
            try:
                # 有些实现为字典 keyed by configuration name
                if cfg_name and hasattr(src, "pool_op_kernel_sizes") and isinstance(src.pool_op_kernel_sizes, dict):
                    v = src.pool_op_kernel_sizes.get(cfg_name, None)
                    if v:
                        candidates.append(v)
                # 也可能直接是 list
                elif hasattr(src, "pool_op_kernel_sizes"):
                    v = src.pool_op_kernel_sizes
                    if v:
                        candidates.append(v)
            except Exception:
                pass
        if candidates:
            return candidates[0]

        # 3) 兜底：根据层数先假设 5 层，然后全部用 [2,2,2]，最后一层 [1,1,1]
        default_stages = 5
        pk = [[2, 2, 2] for _ in range(default_stages)]
        pk[-1] = [1, 1, 1]
        return pk

    def _resolve_conv_kernels(num_stages: int):
        # 1) 常见字段
        if hasattr(cm, "conv_kernel_sizes"):
            ck = cm.conv_kernel_sizes
            if ck is not None and len(ck) > 0:
                return ck
        # 2) 有些版本改名为 kernel_sizes
        if hasattr(cm, "kernel_sizes"):
            ks = cm.kernel_sizes
            if ks is not None and len(ks) > 0:
                return ks
        # 3) 有些版本提供方法
        if hasattr(cm, "get_conv_kernel_sizes"):
            try:
                ks = cm.get_conv_kernel_sizes()
                if ks is not None and len(ks) > 0:
                    return ks
            except Exception:
                pass
        # 4) 兜底：全 3x3x3
        return [[3, 3, 3] for _ in range(num_stages)]

    # ---------- resolve strides & kernels ----------
    strides = _resolve_pool_kernels()
    num_stages = len(strides)

    kernel_sizes = _resolve_conv_kernels(num_stages)
    # 若长度不一致，按最短对齐，或将 kernel_sizes 用最后一个填充到和 strides 一致
    if len(kernel_sizes) < num_stages:
        last = kernel_sizes[-1] if kernel_sizes else [3, 3, 3]
        kernel_sizes = list(kernel_sizes) + [last for _ in range(num_stages - len(kernel_sizes))]
    elif len(kernel_sizes) > num_stages:
        kernel_sizes = kernel_sizes[:num_stages]

    # 维度校验
    dim = len(kernel_sizes[0]) if kernel_sizes and kernel_sizes[0] is not None else 3
    assert dim == 3, "This constructor is for 3D; use the 2D variant for 2D."

    conv_op = convert_dim_to_conv_op(dim)
    label_manager = plans_manager.get_label_manager(dataset_json)

    # ---------- network kwargs ----------
    network_class = MultiEncoderUNet
    kwargs = {
        'conv_bias': False,
        'norm_op': get_matching_instancenorm(conv_op),
        'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
        'dropout_op': None, 'dropout_op_kwargs': None,
        'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
    }
    conv_or_blocks_per_stage = {
        'n_conv_per_stage': getattr(cm, 'n_conv_per_stage_encoder', 2),
        'n_conv_per_stage_decoder': getattr(cm, 'n_conv_per_stage_decoder', 2)
    }

    # ---------- optional text-conditioning via env ----------
    import os
    def _truthy(x: str) -> bool:
        return str(x).lower() in ('1', 'true', 'yes', 'y')

    _text_embed_dim_env = os.environ.get('NNUNET_TEXT_EMBED_DIM', '').strip()
    text_embed_dim = int(_text_embed_dim_env) if _text_embed_dim_env.isdigit() else (None if _text_embed_dim_env == '' else None)
    use_alignment_head = _truthy(os.environ.get('NNUNET_USE_ALIGNMENT_HEAD', '0'))
    return_heatmap = _truthy(os.environ.get('NNUNET_RETURN_HEATMAP', '0'))
    modulation = os.environ.get('NNUNET_TEXT_MODULATION', 'none')  # 'none'|'film'|'gate'
    try:
        heatmap_temperature = float(os.environ.get('NNUNET_HEATMAP_T', '1.0'))
    except Exception:
        heatmap_temperature = 1.0

    # ---------- features_per_stage ----------
    # Prefer reading from architecture kwargs if available, else default progression
    try:
        arch_kwargs = cm.network_arch_init_kwargs
    except Exception:
        arch_kwargs = {}
    feats = arch_kwargs.get('features_per_stage') if isinstance(arch_kwargs, dict) else None
    if feats is None:
        feats = [min(320, 32 * (2 ** i)) for i in range(num_stages)]

    # ---------- instantiate ----------
    model = network_class(
        input_channels=num_input_channels,
        n_stages=num_stages,
        features_per_stage=feats,
        conv_op=conv_op,
        kernel_sizes=kernel_sizes,
        strides=strides,
        num_classes=label_manager.num_segmentation_heads,
        deep_supervision=deep_supervision,
        **conv_or_blocks_per_stage,
        **kwargs,
        # text conditioning (可选参数，模型里如不接受会被忽略或需对应接收)
        text_embed_dim=text_embed_dim,
        use_alignment_head=use_alignment_head,
        return_heatmap=return_heatmap,
        modulation=modulation,
        heatmap_temperature=heatmap_temperature,
    )

    model.apply(InitWeights_He(1e-2))
    return model



def get_multi_encoder_unet_2d_from_plans(
    plans_manager: PlansManager,
    dataset_json: dict,
    configuration_manager: ConfigurationManager,
    num_input_channels: int,
    deep_supervision: bool = True
):
    cm = configuration_manager

    # resolve strides & kernels reusing the 3D helpers
    def _resolve_pool_kernels():
        if hasattr(cm, "pool_op_kernel_sizes"):
            pk = cm.pool_op_kernel_sizes
            if pk is not None and len(pk) > 0:
                return pk
        return [[2, 2] for _ in range(5)] + [[1, 1]]

    def _resolve_conv_kernels(num_stages: int):
        if hasattr(cm, "conv_kernel_sizes"):
            ck = cm.conv_kernel_sizes
            if ck is not None and len(ck) > 0:
                return ck
        if hasattr(cm, "kernel_sizes"):
            ks = cm.kernel_sizes
            if ks is not None and len(ks) > 0:
                return ks
        return [[3, 3] for _ in range(num_stages)]

    strides = _resolve_pool_kernels()
    num_stages = len(strides)
    kernel_sizes = _resolve_conv_kernels(num_stages)
    if len(kernel_sizes) < num_stages:
        last = kernel_sizes[-1]
        kernel_sizes = list(kernel_sizes) + [last] * (num_stages - len(kernel_sizes))
    elif len(kernel_sizes) > num_stages:
        kernel_sizes = kernel_sizes[:num_stages]

    dim = len(kernel_sizes[0])
    assert dim == 2, "This constructor is for 2D; use the 3D variant for 3D."

    conv_op = convert_dim_to_conv_op(dim)
    label_manager = plans_manager.get_label_manager(dataset_json)

    network_class = MultiEncoderUNet
    kwargs = {
        'conv_bias': False,
        'norm_op': get_matching_instancenorm(conv_op),
        'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
        'dropout_op': None, 'dropout_op_kwargs': None,
        'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
    }
    conv_or_blocks_per_stage = {
        'n_conv_per_stage': getattr(cm, 'n_conv_per_stage_encoder', 2),
        'n_conv_per_stage_decoder': getattr(cm, 'n_conv_per_stage_decoder', 2)
    }

    # Optional text-conditioning flags via environment variables (non-breaking defaults)
    import os
    def _truthy(x: str) -> bool:
        return str(x).lower() in ('1', 'true', 'yes', 'y')
    _text_embed_dim_env = os.environ.get('NNUNET_TEXT_EMBED_DIM', '').strip()
    text_embed_dim = int(_text_embed_dim_env) if _text_embed_dim_env.isdigit() else (None if _text_embed_dim_env == '' else None)
    use_alignment_head = _truthy(os.environ.get('NNUNET_USE_ALIGNMENT_HEAD', '0'))
    return_heatmap = _truthy(os.environ.get('NNUNET_RETURN_HEATMAP', '0'))
    modulation = os.environ.get('NNUNET_TEXT_MODULATION', 'none')
    try:
        heatmap_temperature = float(os.environ.get('NNUNET_HEATMAP_T', '1.0'))
    except Exception:
        heatmap_temperature = 1.0

    # features per stage from arch kwargs where possible
    try:
        arch_kwargs = cm.network_arch_init_kwargs
    except Exception:
        arch_kwargs = {}
    feats = arch_kwargs.get('features_per_stage') if isinstance(arch_kwargs, dict) else None
    if feats is None:
        feats = [min(320, 32 * (2 ** i)) for i in range(num_stages)]

    model = network_class(
        input_channels=num_input_channels,
        n_stages=num_stages,
        features_per_stage=feats,
        conv_op=conv_op,
        kernel_sizes=kernel_sizes,
        strides=strides,
        num_classes=label_manager.num_segmentation_heads,
        deep_supervision=deep_supervision,
        **conv_or_blocks_per_stage,
        **kwargs,
        text_embed_dim=text_embed_dim,
        use_alignment_head=use_alignment_head,
        return_heatmap=return_heatmap,
        modulation=modulation,
        heatmap_temperature=heatmap_temperature,
    )
    model.apply(InitWeights_He(1e-2))
    return model
