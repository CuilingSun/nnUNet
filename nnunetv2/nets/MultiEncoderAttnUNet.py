# ==============================
# FILE: MultiEncoderAttnUNet.py
# ==============================
"""
Multi-Encoder + *Controlled* Cross-Attention refinement (decoder-side, residual, gated).

Design goals
------------
1) **Zero-friction drop-in** on top of your existing MultiEncoderUNet implementation.
   - We DO NOT change your encoder/fusion plumbing.
   - We attach a refinement head to the *decoder's highest-resolution stage* using hooks.
2) **Three knobs (τ/α/γ)** for controllable recall vs precision:
   - τ (tau): candidate mask threshold from base logits (only refine where p>τ)
   - α (alpha): residual blend at logits level: seg_final = seg_base + α * (seg_refine - seg_base) * mask
   - γ (gamma): learnable residual gate inside cross-attn head (tanh(γ) ∈ ~[0,1])
3) **Text optional**: works with or without text (if no `text` is provided, refinement is disabled).
4) **No invasive API changes**: we expose a new network `MultiEncoderAttnUNet` and install hooks onto
   the decoder (`decoder.stages` and `decoder.seg_layers[-1]`). If your class names differ slightly,
   please adjust the import below.

How it works
------------
- We capture the last high-resolution decoder feature `x_hi` using a forward hook on the last stage.
- We build a tiny cross-attn refiner that maps `x_hi` + `text` → refined features `x_ref`.
- We create a *sibling* 1x1 conv head that *shares weights* with your original `seg_layers[-1]` so the
  refined branch uses the same classifier.
- In a second forward hook on `seg_layers[-1]`, we blend base logits with refined logits under τ/α.

If you prefer a subclass that overrides `UNetResDecoder.forward`, you can also implement that variant;
this hook-based version avoids assumptions about your decoder constructor signature.
"""
from typing import Optional, Union
import torch
import torch.nn as nn
import os

# ---- Adjust these imports to your repo paths
try:
    from nnunetv2.nets.MultiEncoderUNet import MultiEncoderUNet  # default location
except ImportError:
    try:
        from MultiEncoderUNet import MultiEncoderUNet  # fallback for legacy paths
    except Exception as e:
        raise ImportError("Please ensure MultiEncoderUNet.py is importable: %r" % e)


def _conv_nd_from_op(conv_op: nn.Module) -> int:
    """Return spatial dim (2 or 3) given a conv op class (Conv2d/Conv3d)."""
    name = getattr(conv_op, "__name__", str(conv_op))
    if "3d" in name.lower():
        return 3
    return 2


class TextCrossAttnRefine(nn.Module):
    """Lightweight decoder-side cross-attention refiner with learnable residual gate (tanh(γ)).

    Args:
        in_channels: decoder feature channels at the target (highest-res) stage
        text_dim: dimension of the text embedding vector (D)
        d_model: attention hidden dim (default: min(512, in_channels))
        num_heads: MHA heads
        attn_dropout: attention dropout
        gamma_init: initialize γ so that tanh(γ_init) ≈ gate0 (e.g., 0.2~0.3)
    """
    def __init__(self, in_channels: int, text_dim: int, d_model: Optional[int] = None,
                 num_heads: int = 4, attn_dropout: float = 0.1, gamma_init: float = 0.255):
        super().__init__()
        d = d_model or min(512, max(128, in_channels))
        self.to_d = nn.Conv1d(in_channels, d, kernel_size=1, bias=False)
        self.from_d = nn.Conv1d(d, in_channels, kernel_size=1, bias=False)
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(text_dim, d, bias=False)
        self.v = nn.Linear(text_dim, d, bias=False)
        self.mha = nn.MultiheadAttention(embed_dim=d, num_heads=num_heads, dropout=attn_dropout, batch_first=True)
        # γ: learnable residual gate; we store γ directly and use tanh(γ) in forward
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    @staticmethod
    def _pool_text(text_vec: torch.Tensor) -> torch.Tensor:
        # Accept [B,D] or [B,T,D]; pool to [B,D]
        if text_vec.dim() == 3:
            return text_vec.mean(dim=1)
        return text_vec

    def forward(self, x_hwz: torch.Tensor, text_vec: torch.Tensor) -> torch.Tensor:
        """x_hwz: [B,C,H,W,(Z)]  text_vec: [B,D]
        returns refined features of the same shape as x_hwz
        """
        B, C = x_hwz.shape[:2]
        T = int(torch.tensor(x_hwz.shape[2:]).prod().item())
        x = x_hwz.view(B, C, T).contiguous()                        # [B,C,T]
        x_d = self.to_d(x)                                          # [B,d,T]
        x_d = x_d.transpose(1, 2)                                   # [B,T,d]
        q = self.q(x_d)                                             # [B,T,d]
        kv = self._pool_text(text_vec)
        k = self.k(kv).unsqueeze(1)                                 # [B,1,d]
        v = self.v(kv).unsqueeze(1)                                 # [B,1,d]
        delta, _ = self.mha(q, k, v)                                # [B,T,d]
        delta = delta.transpose(1, 2)                               # [B,d,T]
        delta = self.from_d(delta)                                  # [B,C,T]
        g = torch.tanh(self.gamma)                                  # scalar gate in (-1,1)
        out = x + g * delta                                         # [B,C,T]
        return out.view_as(x_hwz)


class _SegHeadShare(nn.Module):
    """A head that *shares* weights with an existing 1x1 conv head.
    We reference the source head's params so refinement uses the same classifier.
    """
    def __init__(self, src_head: Union[nn.Conv2d, nn.Conv3d]):  # Conv2d or Conv3d
        super().__init__()
        # create a same-shape conv and tie parameters
        Conv = nn.Conv3d if isinstance(src_head, nn.Conv3d) else nn.Conv2d
        self.head = Conv(in_channels=src_head.in_channels,
                         out_channels=src_head.out_channels,
                         kernel_size=src_head.kernel_size,
                         stride=src_head.stride,
                         padding=src_head.padding,
                         bias=src_head.bias is not None)
        # Tie parameters (share tensors)
        self.head.weight = src_head.weight
        if src_head.bias is not None:
            self.head.bias = src_head.bias

    def forward(self, x):
        return self.head(x)


class MultiEncoderAttnUNet(MultiEncoderUNet):
    """Multi-Encoder backbone + *controlled* cross-attention refinement on the decoder's last stage."""

    def __init__(self, *args, use_cross_attn_final: bool = True,
                 cross_alpha: float = 0.25, cross_tau: float = 0.35,
                 cross_attn_stage: str = "penultimate",
                 cross_gamma_init: float = 0.255,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._initialize_cross_attn(use_cross_attn_final=use_cross_attn_final,
                                    cross_alpha=cross_alpha,
                                    cross_tau=cross_tau,
                                    cross_attn_stage=cross_attn_stage,
                                    cross_gamma_init=cross_gamma_init)

    def _initialize_cross_attn(self, *, use_cross_attn_final: bool = True,
                                cross_alpha: float = 0.25, cross_tau: float = 0.35,
                                cross_attn_stage: str = "penultimate",
                                cross_gamma_init: float = 0.255) -> None:
        # Public knobs (modifiable at runtime)
        self.use_cross_attn_final = bool(use_cross_attn_final)
        self.cross_alpha = float(cross_alpha)
        self.cross_tau = float(cross_tau)
        self.cross_attn_stage = str(cross_attn_stage)
        self.cross_gamma_init = float(cross_gamma_init)
        self.fg_index = int(os.environ.get("NNUNET_FG_CHANNEL", "1"))

        # Clean up hooks if we are re-initializing on an existing instance
        if hasattr(self, "_feat_hook_handle") and self._feat_hook_handle is not None:
            try:
                self._feat_hook_handle.remove()
            except Exception:
                pass
        if hasattr(self, "_seg_hook_handle") and self._seg_hook_handle is not None:
            try:
                self._seg_hook_handle.remove()
            except Exception:
                pass

        # placeholders filled during `install_refiner`
        self._hi_feat: Optional[torch.Tensor] = None
        self._text_vec: Optional[torch.Tensor] = None
        self._refiner: Optional[TextCrossAttnRefine] = None
        self._refine_head: Optional[_SegHeadShare] = None
        self._seg_hook_handle = None
        self._feat_hook_handle = None
        self._installed = False

        # Attempt to install immediately (decoder must exist)
        try:
            self.install_refiner()
        except Exception:
            # If decoder plumbing differs, you can call `.install_refiner()` later manually
            pass

    @classmethod
    def from_existing(cls, base_net: MultiEncoderUNet, **kwargs) -> "MultiEncoderAttnUNet":
        """Wrap an already-initialized MultiEncoderUNet instance with cross-attention hooks."""
        if isinstance(base_net, cls):
            base_net._initialize_cross_attn(**kwargs)
            return base_net

        base_net.__class__ = cls  # mutate in place to preserve weights/modules
        base_net._initialize_cross_attn(**kwargs)
        return base_net

    # --- Hook installers ----------------------------------------------------
    def _find_last_stage_module(self) -> nn.Module:
        dec = self.decoder
        # Heuristic: decoder.stages is a ModuleList of per-scale blocks; take the last one.
        stages = getattr(dec, "stages", None)
        if stages is None or len(stages) == 0:
            raise RuntimeError("decoder.stages not found; please adapt accessor here.")
        return stages[-1]

    def _find_seg_head(self) -> nn.Module:
        dec = self.decoder
        seg_layers = getattr(dec, "seg_layers", None)
        if seg_layers is None or len(seg_layers) == 0:
            raise RuntimeError("decoder.seg_layers not found; please adapt accessor here.")
        return seg_layers[-1]

    def install_refiner(self):
        if self._installed:
            return
        # derive dims
        last_stage = self._find_last_stage_module()
        seg_head = self._find_seg_head()
        in_ch = None
        # Try to infer channels from last_stage; fallback to seg_head.in_channels
        for m in reversed(list(last_stage.modules())):
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                in_ch = m.out_channels
                break
        if in_ch is None:
            in_ch = getattr(seg_head, "in_channels", None)
        if in_ch is None:
            raise RuntimeError("Cannot infer decoder channel size for refiner; please set manually.")

        text_dim = getattr(self, "text_embed_dim", None)
        if text_dim is None:
            # If your MultiEncoderUNet stores text dim differently, patch here:
            text_dim = getattr(self, "_text_dim", None)
        if text_dim is None:
            # Without text, we can still install; we will skip refine at runtime when text is None
            text_dim = 256

        # Build modules
        self._refiner = TextCrossAttnRefine(in_channels=in_ch, text_dim=int(text_dim),
                                            d_model=min(512, in_ch), num_heads=4,
                                            attn_dropout=0.1, gamma_init=self.cross_gamma_init)
        self._refine_head = _SegHeadShare(seg_head)

        # Hook 1: capture the last-stage *features*
        def _feat_hook(module: nn.Module, inputs, output):
            if isinstance(output, (tuple, list)):
                output = output[0]
            self._hi_feat = output
            return None

        # Some decoders return tuples; if so, adapt here.
        self._feat_hook_handle = last_stage.register_forward_hook(_feat_hook)

        # Hook 2: blend refined logits into seg head output
        def _seg_hook(module: nn.Module, inputs, output):
            # output is base logits [B,C,H,W,(Z)]
            refiner = self._refiner
            refine_head = self._refine_head
            if (not self.use_cross_attn_final) or refiner is None or refine_head is None:
                return output
            if (self._hi_feat is None) or (self._text_vec is None):
                return output
            alpha = float(self.cross_alpha)
            if alpha == 0.0:
                return output
            tau = float(self.cross_tau)
            with torch.no_grad():
                p = torch.sigmoid(output)
                if output.shape[1] > 1:
                    fg_idx = min(max(int(self.fg_index), 0), output.shape[1] - 1)
                    fg = p[:, fg_idx:fg_idx + 1, ...]
                    mask = (fg > tau).float()
                else:
                    mask = (p > tau).float()
            # refine features
            text = self._text_vec
            hi = self._hi_feat
            if text is None or hi is None:
                return output
            target_device = hi.device
            target_dtype = hi.dtype
            if refiner is not None:
                refiner = refiner.to(device=target_device, dtype=target_dtype)
            if text.device != target_device or text.dtype != target_dtype:
                text = text.to(device=target_device, dtype=target_dtype)
            x_ref = refiner(hi, text)
            seg_ref = refine_head(x_ref)
            return output + alpha * (seg_ref - output) * mask

        self._seg_hook_handle = seg_head.register_forward_hook(_seg_hook)
        self._installed = True

    def remove_refiner(self):
        if self._feat_hook_handle is not None:
            self._feat_hook_handle.remove()
            self._feat_hook_handle = None
        if self._seg_hook_handle is not None:
            self._seg_hook_handle.remove()
            self._seg_hook_handle = None
        self._installed = False

    # --- Forward: we only stash text for hooks to see -----------------------
    def forward(self, x: torch.Tensor, text: Optional[torch.Tensor] = None, **kwargs):
        # Stash text for hooks; if your trainer passes dicts, adapt accordingly
        self._text_vec = text
        return super().forward(x, text=text, **kwargs)
