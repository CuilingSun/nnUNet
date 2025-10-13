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


# ---------- Text Cross-Attention Head ----------
class TextCrossAttentionHead(nn.Module):
    """
    Text-guided cross-attention on the final decoder feature map.

    Q = image tokens (flattened spatial positions)
    K/V = text tokens

    - Input:
        x:        [B, C, H, W] or [B, C, D, H, W]
        text_emb: [B, T, E] or [B, E]  (T = number of text tokens, E = text embedding dim)
        text_mask (optional): [B, T] boolean mask where True indicates valid tokens

    - Output:
        fused feature with the same shape as x

    Notes:
        * Uses gated residual fusion: out = x + g * delta
        * Supports optional text token compression for long prompts
        * Keeps channel dimension C unchanged for easy drop-in before seg_head
    """
    def __init__(
        self,
        in_channels: int,              # decoder's last feature channels (C)
        text_dim: int,                 # text embedding dim (E), e.g., 768 for (Bio)CLIP
        attn_dim: int = 768,           # internal attention dim (d_model); can be == text_dim
        nhead: int = 8,
        dropout: float = 0.0,
        max_text_tokens: Optional[int] = 16,  # compress text tokens if longer than this (None to disable)
        use_ffn: bool = True,
        residual_gate_init: float = 0.1,      # gate init for residual injection strength
    ):
        super().__init__()
        self.in_channels = in_channels
        self.text_dim = text_dim
        self.attn_dim = attn_dim
        self.nhead = nhead
        self.max_text_tokens = max_text_tokens
        self.use_ffn = use_ffn

        # Project image channels -> attention dim, and back
        self.proj_q_in  = nn.Conv1d(in_channels, attn_dim, kernel_size=1, bias=False)
        self.proj_out   = nn.Conv1d(attn_dim, in_channels, kernel_size=1, bias=False)

        # Project text to K/V in the same attention space (handles text_dim != attn_dim)
        self.text_to_k  = nn.Linear(text_dim, attn_dim, bias=False)
        self.text_to_v  = nn.Linear(text_dim, attn_dim, bias=False)

        # Multi-head attention (batch_first=False to be robust across torch versions)
        self.attn = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=False
        )

        # Normalization + optional FFN (Transformer block style)
        self.ln_q   = nn.LayerNorm(attn_dim)
        self.ln_out = nn.LayerNorm(attn_dim)

        if use_ffn:
            self.ffn = nn.Sequential(
                nn.Linear(attn_dim, 4 * attn_dim),
                nn.GELU(),
                nn.Linear(4 * attn_dim, attn_dim),
            )
        else:
            self.ffn = nn.Identity()

        # Learnable gate controls residual strength; keeps early training close to baseline
        residual_gate_init = float(residual_gate_init)
        residual_gate_init = min(max(residual_gate_init, 1e-4), 1 - 1e-4)
        gate_logit = np.log(residual_gate_init / (1.0 - residual_gate_init))
        self.residual_gate_logit = nn.Parameter(torch.tensor(gate_logit, dtype=torch.float32))
        self._last_gate: Optional[torch.Tensor] = None

    @staticmethod
    def _flatten_spatial(x: torch.Tensor):
        """
        Flatten spatial dims to a token dimension.
        - 2D: [B, C, H, W] -> [B, C, N]
        - 3D: [B, C, D, H, W] -> [B, C, N]
        """
        B, C = x.shape[:2]
        spatial = x.shape[2:]
        N = 1
        for s in spatial:
            N *= s
        return x.reshape(B, C, N), spatial

    @staticmethod
    def _unflatten_spatial(x_tokens: torch.Tensor, spatial: tuple):
        """
        Restore flattened tokens to spatial shape.
        - x_tokens: [B, C, N]
        - returns:  [B, C, *spatial]
        """
        B, C, _ = x_tokens.shape
        return x_tokens.reshape(B, C, *spatial)

    def _maybe_compress_text(self, text_emb: torch.Tensor, text_mask: Optional[torch.Tensor]):
        """
        Optionally compress long text sequences to at most `max_text_tokens`.
        Strategies:
            - If T <= max_text_tokens: do nothing
            - Else: mean-pool into `max_text_tokens` chunks (mask-aware)
        """
        if self.max_text_tokens is None:
            return text_emb, text_mask

        B, T, E = text_emb.shape
        if T <= self.max_text_tokens:
            return text_emb, text_mask

        # Compute chunk size and mean-pool along token dim
        # Example: T=77, max=16 -> ~16 chunks
        chunk = (T + self.max_text_tokens - 1) // self.max_text_tokens  # ceil
        # Pad to multiple of chunk
        pad_len = (chunk - (T % chunk)) % chunk
        if pad_len > 0:
            pad = torch.zeros(B, pad_len, E, dtype=text_emb.dtype, device=text_emb.device)
            text_emb = torch.cat([text_emb, pad], dim=1)
            if text_mask is not None:
                mask_pad = torch.zeros(B, pad_len, dtype=text_mask.dtype, device=text_mask.device)
                text_mask = torch.cat([text_mask, mask_pad], dim=1)
            T = text_emb.shape[1]

        # Reshape to [B, num_groups, chunk, E] and mean over chunk
        num_groups = T // chunk
        text_emb = text_emb.view(B, num_groups, chunk, E)
        text_emb = text_emb.mean(dim=2)  # [B, num_groups, E]

        if text_mask is not None:
            text_mask = text_mask.view(B, num_groups, chunk)
            # A group is valid if any token in the chunk is valid
            text_mask = text_mask.any(dim=2)  # [B, num_groups]

        return text_emb, text_mask

    def forward(
        self,
        x: torch.Tensor,                      # [B, C, H, W] or [B, C, D, H, W]
        text_emb: Optional[torch.Tensor],     # [B, T, E] or [B, E]
        text_mask: Optional[torch.Tensor] = None,  # [B, T] (True for valid tokens)
    ) -> torch.Tensor:
        # Early exit if no text provided
        if text_emb is None:
            return x

        # Ensure text is [B, T, E]
        if text_emb.dim() == 2:  # [B, E] -> [B, 1, E]
            text_emb = text_emb.unsqueeze(1)

        # Optional compression for long prompts
        text_emb, text_mask = self._maybe_compress_text(text_emb, text_mask)

        B, T, E = text_emb.shape
        # Flatten image spatial dims -> tokens
        x_tokens, spatial = self._flatten_spatial(x)          # [B, C, N]

        # Project image tokens to attention space (d_model)
        q = self.proj_q_in(x_tokens)                          # [B, d_model, N]
        q = q.permute(2, 0, 1)                                # [N, B, d_model]
        q = self.ln_q(q)

        # Prepare K/V from text
        k = self.text_to_k(text_emb)                          # [B, T, d_model]
        v = self.text_to_v(text_emb)                          # [B, T, d_model]
        k = k.permute(1, 0, 2)                                # [T, B, d_model]
        v = v.permute(1, 0, 2)                                # [T, B, d_model]

        # Build key padding mask (True means to ignore)
        key_padding_mask = None
        if text_mask is not None:
            # text_mask: True for valid -> MultiheadAttention expects True for "to be ignored"
            key_padding_mask = ~text_mask.bool()              # [B, T]

        # Cross-Attention: Q=image, K/V=text
        attn_out, _ = self.attn(
            q, k, v,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )                                                     # [N, B, d_model]

        # Post-attention block with residual + FFN
        z = self.ln_out(q + attn_out)                         # [N, B, d_model]
        z = z + self.ffn(z)                                   # [N, B, d_model]

        # Back to [B, d_model, N] -> [B, C, N]
        z = z.permute(1, 2, 0).contiguous()                   # [B, d_model, N]
        delta = self.proj_out(z)                              # [B, C, N]

        gate = torch.sigmoid(self.residual_gate_logit)        # scalar in (0,1)
        if gate.dim() == 0:
            gate = gate.view(1, 1, 1)
        self._last_gate = gate.detach()

        # Residual fusion and reshape to spatial
        out = x_tokens + gate * delta                         # [B, C, N]
        out = self._unflatten_spatial(out, spatial)           # [B, C, *spatial]
        return out


# ---------- Upsample Layer ----------
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
    """ 3x3 (x3) conv fusion, projecting back to per-modality channels."""
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
    
# ---------- MoE Fusion: Modality-as-Experts ----------
class ReliabilityModalityGateGlobal(nn.Module):
    """
    Modality-as-Experts global MoE fusion.
    Each modality branch is treated as an expert.
    A router computes global weights per modality and mixes them.

    Inputs:
        feats: list of tensors [f_t2, f_dwi, f_adc],
               each of shape [B, C, D, H, W] (same channels & spatial size)
    Outputs:
        fused: tensor [B, C, D, H, W]
    """

    def __init__(
        self,
        conv_op: Type[_ConvNd],
        channels_per_modality: int,  # feature channels per modality
        num_modalities: int,         # number of modalities (e.g., 3)
        norm_op: Union[None, Type[nn.Module]],
        norm_op_kwargs: dict,
        nonlin: Union[None, Type[nn.Module]],
        nonlin_kwargs: dict,
        conv_bias: bool = False,
        hidden: Optional[int] = None,
        temperature: float = 1.2
    ):
        super().__init__()
        self.M = num_modalities
        self.C = channels_per_modality
        self.temperature = temperature

        # Global pooling: compress concatenated features to [B, hid]
        hid = hidden or max(self.C // 4, 8)
        self.img_proj = nn.Sequential(
            conv_op(self.M * self.C, hid, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1) if conv_op == nn.Conv3d else nn.AdaptiveAvgPool2d(1)
        )

        # Router: linear layer produces modality scores -> softmax -> gates
        self.fc = nn.Linear(hid, self.M)
        nn.init.zeros_(self.fc.bias)

        # Align each modality with a 1x1 conv before mixing
        self.align = nn.ModuleList([
            conv_op(self.C, self.C, kernel_size=1, bias=False)
            for _ in range(self.M)
        ])

        # Normalization and activation after fusion
        self.post_norm = (
            norm_op(self.C, **(norm_op_kwargs or {}))
            if norm_op is not None else nn.Identity()
        )
        self.post_nonlin = (
            nonlin(**(nonlin_kwargs or {}))
            if nonlin is not None else nn.Identity()
        )

    def forward(self, feats: List[torch.Tensor], return_gates: bool = False):
        assert len(feats) == self.M
        # Concatenate modality features -> [B, M*C, ...]
        x = torch.cat(feats, dim=1)
        # Project to global vector -> [B, hid]
        h = self.img_proj(x).flatten(1)
        # Compute modality gates
        logits = self.fc(h) / self.temperature
        gates = torch.softmax(logits, dim=-1)  # [B, M]

        # Align each modality feature with 1x1 conv
        aligned = [a(f) for a, f in zip(self.align, feats)]  # list of [B, C, ...]
        # Broadcast gates to match spatial dims
        while gates.dim() < aligned[0].dim() + 1:
            gates = gates.unsqueeze(-1)
        # Weighted sum across modalities
        fused = sum(gates[:, i] * aligned[i] for i in range(self.M))

        fused = self.post_nonlin(self.post_norm(fused))
        if return_gates:
            return fused, gates  # [B,C,...], [B,M]
        return fused

#---------- MoE Fusion Block Wrapper ----------
class FusionBlockMoEAdapter(nn.Module):
    """
    Drop-in replacement for your original FusionBlock.
    Same API: forward(cat) where cat=[B, M*C, ...].
    Splits cat into modality features and applies a modality-level MoE gate.
    Logs the last gates in self._last_gates for visualization.
    """
    def __init__(self, *, conv_op, num_modalities: int, channels_per_modality: int,
                 norm_op=None, norm_op_kwargs=None, nonlin=None, nonlin_kwargs=None,
                 temperature: float = 1.2):
        super().__init__()
        self.M = num_modalities
        self.C = channels_per_modality
        self._last_gates: Optional[torch.Tensor] = None  # [B, M] gates of last forward

        # Core MoE gate
        self.gate = ReliabilityModalityGateGlobal(
            conv_op=conv_op,
            channels_per_modality=self.C,
            num_modalities=self.M,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs or {},
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs or {},
            temperature=temperature
        )

    def forward(self, cat_feats: torch.Tensor) -> torch.Tensor:
        # Split concatenated input back into modalities
        feats = torch.chunk(cat_feats, chunks=self.M, dim=1)  # list of [B, C, ...]

        # Call the underlying gate
        fused, gates = self.gate(feats, return_gates=True)  # <-- modify gate to return gates
        self._last_gates = gates.detach()                   # save for logging

        return fused

# ---------- Encoder ----------
class UNetResEncoder(nn.Module):
    """return_skips=True:return features from shallow to deep as a list"""
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

# ---------Decoder ----------
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
        # Highest-resolution decoder feature map (used by downstream heads)
        self.out_channels = encoder.output_channels[0]

    def forward(self, skips: List[torch.Tensor], text: Optional[torch.Tensor] = None,
                return_final_feature: bool = False):
        lres_input = skips[-1]
        seg_outputs: List[torch.Tensor] = []
        final_feature: Optional[torch.Tensor] = None

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

            if s == len(self.stages) - 1:
                final_feature = x

            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))

            lres_input = x

        seg_outputs = seg_outputs[::-1]
        seg_result = seg_outputs if self.deep_supervision else seg_outputs[0]

        if return_final_feature:
            return seg_result, final_feature

        return seg_result

    def compute_conv_feature_map_size(self, input_size):
        return np.int64(0)


# ---------- Multi-encoder core ----------
class MultiEncoderUNet(nn.Module):
    """
    Multiple encoders (one per modality) + stage-wise fusion + a shared decoder.
    Input: [B, M, ...], where M is the number of modalities (from num_input_channels).
    """
    def __init__(
        self,
        input_channels: int,   # = number of modalities
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
        dropout_op: Union[None, Type[_DropoutNd]] = None,   # placeholder
        dropout_op_kwargs: dict = None,                     # placeholder
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
        moe_temperature: float = 1.2,
        use_cross_attn_final: bool = False,
        use_text_adapter: bool = False,
    ):
        super().__init__()
        num_modalities = input_channels
        self.text_embed_dim = text_embed_dim
        self.modulation = modulation
        self.use_alignment_head = use_alignment_head
        self.return_heatmap = return_heatmap
        self.heatmap_temperature = heatmap_temperature
        self.moe_temperature = moe_temperature
        self.use_cross_attn_final = use_cross_attn_final
        self.use_text_adapter = bool(use_text_adapter) and self.text_embed_dim is not None
        self.deep_supervision = deep_supervision

        if self.use_text_adapter:
            self.text_adapter = nn.Sequential(
                nn.LayerNorm(self.text_embed_dim),
                nn.Linear(self.text_embed_dim, self.text_embed_dim, bias=False),
            )
        else:
            self.text_adapter = nn.Identity()

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

        # Bottleneck fusion
        bottleneck_features_out = self.encoders[0].output_channels[-1]
        self.bottleneck_fusion = FusionBlockMoEAdapter(
            conv_op=conv_op,
            num_modalities=num_modalities,
            channels_per_modality=bottleneck_features_out,
            norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin, nonlin_kwargs=nonlin_kwargs,
            temperature=self.moe_temperature,
        )

        # Skip fusion blocks
        self.skip_fusions = nn.ModuleList()
        for i in range(n_stages - 1):
            skip_out = self.encoders[0].output_channels[i]
            self.skip_fusions.append(
                FusionBlockMoEAdapter(
                    conv_op=conv_op,
                    num_modalities=num_modalities,
                    channels_per_modality=skip_out,
                    norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                    nonlin=nonlin, nonlin_kwargs=nonlin_kwargs,
                    temperature=self.moe_temperature,
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

        dec_out_ch = getattr(self.decoder, 'out_channels', None)
        if self.use_cross_attn_final and self.text_embed_dim is not None and dec_out_ch is not None:
            self.cross_attn_final = TextCrossAttentionHead(
                in_channels=dec_out_ch,
                text_dim=self.text_embed_dim,
                attn_dim=self.text_embed_dim,
                nhead=8,
                dropout=0.0,
                max_text_tokens=16,
                use_ffn=True,
            )
        else:
            self.cross_attn_final = None

    def forward(self, x, text: Optional[torch.Tensor] = None,
                return_extra: bool = False) -> Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]:
        modalities = list(x.split(1, dim=1))  # M x [B,1,...]
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

        need_final_feature = self.cross_attn_final is not None
        if text is not None and self.use_text_adapter:
            text = self.text_adapter(text)

        decoder_out = self.decoder(
            fused_skips,
            text=text,
            return_final_feature=need_final_feature,
        )

        if need_final_feature:
            seg, final_feature = decoder_out
            if text is not None:
                refined_feature = self.cross_attn_final(final_feature, text)
                refined_logits = self.decoder.seg_layers[-1](refined_feature)

                if self.deep_supervision:
                    seg = list(seg)
                    seg[0] = refined_logits
                else:
                    seg = refined_logits
        else:
            seg = decoder_out

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


# ---------- from_plans entry ----------
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
    Compatibility across ConfigurationManager versions:
      - Kernel sizes: prefer cm.conv_kernel_sizes / cm.kernel_sizes / cm.get_conv_kernel_sizes().
                   If none exist, fall back to [3,3,3] per stage.
      - Downsampling strides (pool_op_kernel_sizes): prefer cm.pool_op_kernel_sizes, 
                   otherwise fallback to plans_manager or cm.plans_manager, 
                   if still missing, use [[2,2,2], ...] (last level [1,1,1]).
    """
    cm = configuration_manager

    # ---------- helpers ----------
    def _resolve_pool_kernels():
        # 1) Direct attribute (common across versions)
        if hasattr(cm, "pool_op_kernel_sizes"):
            pk = cm.pool_op_kernel_sizes
            if pk is not None and len(pk) > 0:
                return pk

        # 2) Fallback via plans_manager / configuration name
        cfg_name = getattr(cm, "configuration_name", None)
        pm = getattr(cm, "plans_manager", None)
        candidates = []
        for src in (pm, plans_manager):
            if src is None:
                continue
            try:
                # Some implementations use a dict keyed by configuration name.
                if cfg_name and hasattr(src, "pool_op_kernel_sizes") and isinstance(src.pool_op_kernel_sizes, dict):
                    v = src.pool_op_kernel_sizes.get(cfg_name, None)
                    if v:
                        candidates.append(v)
                # It could also be a plain list.
                elif hasattr(src, "pool_op_kernel_sizes"):
                    v = src.pool_op_kernel_sizes
                    if v:
                        candidates.append(v)
            except Exception:
                pass
        if candidates:
            return candidates[0]

        # 3) Fallback: assume 5 levels; use [2,2,2] for all, with [1,1,1] at the last level.
        default_stages = 5
        pk = [[2, 2, 2] for _ in range(default_stages)]
        pk[-1] = [1, 1, 1]
        return pk

    def _resolve_conv_kernels(num_stages: int):
        # 1) Common attribute
        if hasattr(cm, "conv_kernel_sizes"):
            ck = cm.conv_kernel_sizes
            if ck is not None and len(ck) > 0:
                return ck
        # 2) Some versions rename it to kernel_sizes.
        if hasattr(cm, "kernel_sizes"):
            ks = cm.kernel_sizes
            if ks is not None and len(ks) > 0:
                return ks
        # 3) Some versions provide a method.
        if hasattr(cm, "get_conv_kernel_sizes"):
            try:
                ks = cm.get_conv_kernel_sizes()
                if ks is not None and len(ks) > 0:
                    return ks
            except Exception:
                pass
        # 4) Fallback: all 3x3x3
        return [[3, 3, 3] for _ in range(num_stages)]

    # ---------- resolve strides & kernels ----------
    strides = _resolve_pool_kernels()
    num_stages = len(strides)

    kernel_sizes = _resolve_conv_kernels(num_stages)
    # If lengths differ, align to the shorter one, or pad kernel_sizes with the last value to match strides.
    if len(kernel_sizes) < num_stages:
        last = kernel_sizes[-1] if kernel_sizes else [3, 3, 3]
        kernel_sizes = list(kernel_sizes) + [last for _ in range(num_stages - len(kernel_sizes))]
    elif len(kernel_sizes) > num_stages:
        kernel_sizes = kernel_sizes[:num_stages]

    # Dimensionality check
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
    use_cross_attn_final = _truthy(os.environ.get('NNUNET_USE_CROSS_ATTN_FINAL', '0'))
    use_text_adapter = _truthy(os.environ.get('NNUNET_USE_TEXT_ADAPTOR', '0'))
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
        # text conditioning (optional; ignored if the model does not accept it)
        text_embed_dim=text_embed_dim,
        use_alignment_head=use_alignment_head,
        return_heatmap=return_heatmap,
        modulation=modulation,
        heatmap_temperature=heatmap_temperature,
        use_cross_attn_final=use_cross_attn_final,
        use_text_adapter=use_text_adapter,
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
    use_cross_attn_final = _truthy(os.environ.get('NNUNET_USE_CROSS_ATTN_FINAL', '0'))
    use_text_adapter = _truthy(os.environ.get('NNUNET_USE_TEXT_ADAPTOR', '0'))
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
        use_cross_attn_final=use_cross_attn_final,
        use_text_adapter=use_text_adapter,
    )
    model.apply(InitWeights_He(1e-2))
    return model
