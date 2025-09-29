"""Multi-encoder variant that mirrors the nnU-Net PlainConvUNet backbone.

This network instantiates one PlainConv encoder per modality, fuses their skip
features, and shares a PlainConv-style decoder with optional text modulation and
alignment/heatmap heads. Shapes and module names mimic the original
PlainConvUNet so that pretrained encoder weights from vanilla nnU-Net
(`encoder.*`) can be mapped onto `encoders.0.*`.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from dynamic_network_architectures.building_blocks.helper import (
    convert_dim_to_conv_op,
    convert_conv_op_to_dim,
    get_matching_instancenorm,
    maybe_convert_scalar_to_list,
    get_matching_convtransp,
)
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.initialization.weight_init import InitWeights_He

from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager


class FusionBlock(nn.Module):
    """Fuse features from all encoders back to single-encoder channel width."""

    def __init__(
        self,
        conv_op: type,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, ...],
        padding: Tuple[int, ...],
        norm_op: Optional[type],
        norm_op_kwargs: Optional[dict],
        nonlin: Optional[type],
        nonlin_kwargs: Optional[dict],
        conv_bias: bool = False,
    ) -> None:
        super().__init__()
        ks = kernel_size if isinstance(kernel_size, tuple) else tuple(kernel_size)
        pad = padding if isinstance(padding, tuple) else tuple(padding)
        modules: List[nn.Module] = [
            conv_op(in_channels, out_channels, kernel_size=ks, padding=pad, bias=conv_bias),
        ]
        if norm_op is not None:
            modules.append(norm_op(out_channels, **(norm_op_kwargs or {})))
        if nonlin is not None:
            modules.append(nonlin(**(nonlin_kwargs or {})))
        self.block = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.block(x)


class UNetDecoderWithText(nn.Module):
    """PlainConv decoder with optional per-stage text modulation (FiLM/Gate)."""

    def __init__(
        self,
        encoder: PlainConvEncoder,
        num_classes: int,
        n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
        deep_supervision: bool,
        nonlin_first: bool = False,
        text_embed_dim: Optional[int] = None,
        modulation: str = 'none',
    ) -> None:
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

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)

        stages: List[nn.Module] = []
        transpconvs: List[nn.Module] = []
        seg_layers: List[nn.Module] = []
        modulators: List[nn.Module] = []

        for s in range(1, n_stages_encoder):
            ch_below = encoder.output_channels[-s]
            ch_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]

            transpconvs.append(
                transpconv_op(
                    ch_below,
                    ch_skip,
                    stride_for_transpconv,
                    stride_for_transpconv,
                    bias=encoder.conv_bias,
                )
            )

            stages.append(
                StackedConvBlocks(
                    n_conv_per_stage[s - 1],
                    encoder.conv_op,
                    2 * ch_skip,
                    ch_skip,
                    encoder.kernel_sizes[-(s + 1)],
                    1,
                    encoder.conv_bias,
                    encoder.norm_op,
                    encoder.norm_op_kwargs,
                    encoder.dropout_op,
                    encoder.dropout_op_kwargs,
                    encoder.nonlin,
                    encoder.nonlin_kwargs,
                    nonlin_first,
                )
            )

            seg_layers.append(encoder.conv_op(ch_skip, num_classes, 1, 1, 0, bias=True))

            if text_embed_dim is not None and modulation in ('film', 'gate'):
                hidden = max(32, ch_skip // 2)
                if modulation == 'film':
                    modulators.append(
                        nn.Sequential(
                            nn.Linear(text_embed_dim, hidden),
                            nn.ReLU(inplace=True),
                            nn.Linear(hidden, 2 * ch_skip),
                        )
                    )
                else:
                    modulators.append(
                        nn.Sequential(
                            nn.Linear(text_embed_dim, hidden),
                            nn.ReLU(inplace=True),
                            nn.Linear(hidden, ch_skip),
                        )
                    )
            else:
                modulators.append(nn.Identity())

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)
        self.modulators = nn.ModuleList(modulators)

    def forward(self, skips: List[torch.Tensor], text: Optional[torch.Tensor] = None):
        lres_input = skips[-1]
        seg_outputs: List[torch.Tensor] = []

        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s + 2)]), 1)
            x = self.stages[s](x)

            mod = self.modulators[s]
            if not isinstance(mod, nn.Identity):
                assert text is not None, 'Text embedding must be provided when modulation is enabled.'
                params = mod(text)
                b, c = x.shape[0], x.shape[1]
                spatial = [1] * (x.ndim - 2)
                if self.modulation == 'film':
                    gamma, beta = params.chunk(2, dim=-1)
                    x = gamma.view(b, c, *spatial) * x + beta.view(b, c, *spatial)
                else:
                    alpha = torch.sigmoid(params).view(b, c, *spatial)
                    x = alpha * x

            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == len(self.stages) - 1:
                seg_outputs.append(self.seg_layers[-1](x))

            lres_input = x

        seg_outputs = seg_outputs[::-1]
        return seg_outputs if self.deep_supervision else seg_outputs[0]


class MultiEncodernnUNet(nn.Module):
    """Multi-modal encoders + PlainConv-style decoder with text/alignment heads."""

    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: type,
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
        num_classes: int,
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
        conv_bias: bool = False,
        norm_op: Optional[type] = None,
        norm_op_kwargs: Optional[dict] = None,
        dropout_op: Optional[type] = None,
        dropout_op_kwargs: Optional[dict] = None,
        nonlin: Optional[type] = None,
        nonlin_kwargs: Optional[dict] = None,
        deep_supervision: bool = False,
        nonlin_first: bool = False,
        text_embed_dim: Optional[int] = None,
        use_alignment_head: bool = False,
        return_heatmap: bool = False,
        heatmap_temperature: float = 1.0,
        modulation: str = 'none',
    ) -> None:
        super().__init__()
        self.num_modalities = input_channels
        self.text_embed_dim = text_embed_dim
        self.modulation = modulation
        self.use_alignment_head = use_alignment_head
        self.return_heatmap = return_heatmap
        self.heatmap_temperature = heatmap_temperature
        self.deep_supervision = deep_supervision

        if isinstance(kernel_sizes, int):
            kernel_sizes = [maybe_convert_scalar_to_list(conv_op, kernel_sizes) for _ in range(n_stages)]
        elif isinstance(kernel_sizes, (tuple, list)) and isinstance(kernel_sizes[0], int):
            kernel_sizes = [maybe_convert_scalar_to_list(conv_op, k) for k in kernel_sizes]
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(kernel_sizes) == n_stages
        assert len(strides) == n_stages
        assert len(n_conv_per_stage) == n_stages
        assert len(n_conv_per_stage_decoder) == (n_stages - 1)

        self.encoders = nn.ModuleList([
            PlainConvEncoder(
                input_channels=1,
                n_stages=n_stages,
                features_per_stage=features_per_stage,
                conv_op=conv_op,
                kernel_sizes=kernel_sizes,
                strides=strides,
                n_conv_per_stage=n_conv_per_stage,
                conv_bias=conv_bias,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                dropout_op=dropout_op,
                dropout_op_kwargs=dropout_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
                return_skips=True,
                nonlin_first=nonlin_first,
            )
            for _ in range(self.num_modalities)
        ])

        # fusion blocks for each skip level (including bottleneck)
        self.skip_fusions = nn.ModuleList()
        for stage_idx in range(n_stages):
            in_ch = self.encoders[0].output_channels[stage_idx] * self.num_modalities
            out_ch = self.encoders[0].output_channels[stage_idx]
            k = tuple(kernel_sizes[stage_idx])
            pad = tuple(np.array(k) // 2)
            self.skip_fusions.append(
                FusionBlock(
                    conv_op=conv_op,
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=k,
                    padding=pad,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                    conv_bias=conv_bias,
                )
            )

        self.decoder = UNetDecoderWithText(
            encoder=self.encoders[0],
            num_classes=num_classes,
            n_conv_per_stage=n_conv_per_stage_decoder,
            deep_supervision=deep_supervision,
            nonlin_first=nonlin_first,
            text_embed_dim=text_embed_dim,
            modulation=modulation,
        )

        if self.use_alignment_head and self.text_embed_dim is not None:
            bottleneck_channels = self.encoders[0].output_channels[-1]
            dim = convert_conv_op_to_dim(conv_op)
            k1 = (1,) * dim
            self.align_conv = conv_op(bottleneck_channels, self.text_embed_dim, kernel_size=k1, bias=True)
            self.text_proj = nn.Linear(self.text_embed_dim, self.text_embed_dim, bias=False)
        else:
            self.align_conv = None
            self.text_proj = None

    def forward(
        self,
        x: torch.Tensor,
        text: Optional[torch.Tensor] = None,
        return_extra: bool = False,
    ) -> Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]:
        modalities = list(x.split(1, dim=1))  # -> M × [B,1,...]
        all_skips = [enc(mod) for enc, mod in zip(self.encoders, modalities)]

        fused_skips: List[torch.Tensor] = []
        for stage_idx in range(len(all_skips[0])):
            cat_stage = torch.cat([sk[stage_idx] for sk in all_skips], dim=1)
            fused = self.skip_fusions[stage_idx](cat_stage)
            fused_skips.append(fused)

        seg = self.decoder(fused_skips, text=text)

        if self.use_alignment_head and self.align_conv is not None and text is not None:
            bottleneck = fused_skips[-1]
            feat = self.align_conv(bottleneck)
            feat = F.normalize(feat, dim=1, eps=1e-6)
            t = self.text_proj(text)
            t = F.normalize(t, dim=-1, eps=1e-6)
            view_shape = [t.shape[0], t.shape[1]] + [1] * (feat.ndim - 2)
            t_view = t.view(*view_shape)
            sim = (feat * t_view).sum(dim=1, keepdim=True)
            if self.return_heatmap:
                if isinstance(seg, list):
                    target_spatial = seg[0].shape[2:]
                else:
                    target_spatial = seg.shape[2:]
                sim = F.interpolate(
                    sim,
                    size=target_spatial,
                    mode='trilinear' if len(target_spatial) == 3 else 'bilinear',
                    align_corners=False,
                )
            sim = torch.sigmoid(sim / max(1e-6, self.heatmap_temperature))
            if return_extra:
                return {'seg': seg, 'sim': sim}
        return seg

    def compute_conv_feature_map_size(self, input_size):
        return np.int64(0)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)


def _resolve_pool_kernels(cfg: ConfigurationManager, plans_manager: PlansManager) -> List[List[int]]:
    if hasattr(cfg, 'pool_op_kernel_sizes') and cfg.pool_op_kernel_sizes:
        return cfg.pool_op_kernel_sizes
    cfg_name = getattr(cfg, 'configuration_name', None)
    for source in (cfg.plans_manager if hasattr(cfg, 'plans_manager') else None, plans_manager):
        if source is None:
            continue
        pool = getattr(source, 'pool_op_kernel_sizes', None)
        if isinstance(pool, dict) and cfg_name:
            val = pool.get(cfg_name)
            if val:
                return val
        if isinstance(pool, list) and pool:
            return pool
    # fallback: 5 stages of [2,2,(2)] with last [1]
    dim = len(cfg.patch_size)
    default = [[2] * dim for _ in range(5)]
    default[-1] = [1] * dim
    return default


def _resolve_conv_kernels(cfg: ConfigurationManager, num_stages: int) -> List[List[int]]:
    if hasattr(cfg, 'conv_kernel_sizes') and cfg.conv_kernel_sizes:
        return cfg.conv_kernel_sizes
    if hasattr(cfg, 'kernel_sizes') and cfg.kernel_sizes:
        return cfg.kernel_sizes
    if hasattr(cfg, 'get_conv_kernel_sizes'):
        try:
            ks = cfg.get_conv_kernel_sizes()
            if ks:
                return ks
        except Exception:
            pass
    dim = len(cfg.patch_size)
    return [[3] * dim for _ in range(num_stages)]


def _truthy(val: str) -> bool:
    return str(val).lower() in ('1', 'true', 'yes', 'y')


def build_multi_encoder_from_plans(
    plans_manager: PlansManager,
    dataset_json: dict,
    configuration_manager: ConfigurationManager,
    num_input_channels: int,
    deep_supervision: bool = True,
) -> MultiEncodernnUNet:
    strides = _resolve_pool_kernels(configuration_manager, plans_manager)
    num_stages = len(strides)
    kernel_sizes = _resolve_conv_kernels(configuration_manager, num_stages)
    if len(kernel_sizes) < num_stages:
        kernel_sizes = list(kernel_sizes) + [kernel_sizes[-1]] * (num_stages - len(kernel_sizes))
    elif len(kernel_sizes) > num_stages:
        kernel_sizes = kernel_sizes[:num_stages]

    dim = len(kernel_sizes[0])
    conv_op = convert_dim_to_conv_op(dim)
    norm_op = get_matching_instancenorm(conv_op)
    label_manager = plans_manager.get_label_manager(dataset_json)

    try:
        arch_kwargs = configuration_manager.network_arch_init_kwargs
    except Exception:
        arch_kwargs = {}

    feats = None
    if isinstance(arch_kwargs, dict):
        feats = arch_kwargs.get('features_per_stage') or arch_kwargs.get('features_per_stage_encoder')
    if feats is None:
        try:
            feats = configuration_manager.network_arch_init_kwargs['features_per_stage']
        except Exception:
            base = getattr(configuration_manager, 'UNet_base_num_features', 32)
            max_features = getattr(configuration_manager, 'unet_max_num_features', 320)
            feats = [min(max_features, base * (2 ** i)) for i in range(num_stages)]

    n_conv_per_stage_encoder = getattr(configuration_manager, 'n_conv_per_stage_encoder', 2)
    n_conv_per_stage_decoder = getattr(configuration_manager, 'n_conv_per_stage_decoder', 2)
    conv_bias = False
    nonlin = nn.LeakyReLU
    nonlin_kwargs = {'inplace': True}

    text_embed_dim_env = os.environ.get('NNUNET_TEXT_EMBED_DIM', '').strip()
    text_embed_dim = int(text_embed_dim_env) if text_embed_dim_env.isdigit() else None
    use_alignment_head = _truthy(os.environ.get('NNUNET_USE_ALIGNMENT_HEAD', '0'))
    return_heatmap = _truthy(os.environ.get('NNUNET_RETURN_HEATMAP', '0'))
    modulation = os.environ.get('NNUNET_TEXT_MODULATION', 'none')
    try:
        heatmap_temperature = float(os.environ.get('NNUNET_HEATMAP_T', '1.0'))
    except Exception:
        heatmap_temperature = 1.0

    network = MultiEncodernnUNet(
        input_channels=num_input_channels,
        n_stages=num_stages,
        features_per_stage=feats,
        conv_op=conv_op,
        kernel_sizes=kernel_sizes,
        strides=strides,
        n_conv_per_stage=n_conv_per_stage_encoder,
        num_classes=label_manager.num_segmentation_heads,
        n_conv_per_stage_decoder=n_conv_per_stage_decoder,
        conv_bias=conv_bias,
        norm_op=norm_op,
        norm_op_kwargs={'eps': 1e-5, 'affine': True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nonlin,
        nonlin_kwargs=nonlin_kwargs,
        deep_supervision=deep_supervision,
        nonlin_first=False,
        text_embed_dim=text_embed_dim,
        use_alignment_head=use_alignment_head,
        return_heatmap=return_heatmap,
        heatmap_temperature=heatmap_temperature,
        modulation=modulation,
    )
    network.apply(InitWeights_He(1e-2))
    return network


def get_multi_encoder_from_plans(
    plans_manager: PlansManager,
    dataset_json: dict,
    configuration_manager: ConfigurationManager,
    num_input_channels: int,
    deep_supervision: bool = True,
) -> MultiEncodernnUNet:
    return build_multi_encoder_from_plans(
        plans_manager,
        dataset_json,
        configuration_manager,
        num_input_channels,
        deep_supervision=deep_supervision,
    )
