"""Text-conditioned trainer for the MultiEncodernnUNet backbone.

This extends the fusion-style multi-encoder trainer with:
  * optional text embeddings (fixed from env or learnable fallback)
  * auxiliary alignment/heatmap losses like the MultiEncoderUNetText trainer
  * robust loading of pretrained nnUNetv2 encoder checkpoints for every encoder

Environment toggles mirror the existing text trainer so scripts remain compatible.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMultiEncodernnUNet import (
    nnUNetTrainerMultiEncodernnUNet,
)

PREFIX_CANDIDATES: Tuple[str, ...] = (
    "module.",
    "network.",
    "model.",
    "state_dict.",
    "nnunet.",
    "seg_network.",
)


def _strip_common_prefix(state: Dict[str, torch.Tensor], prefixes: Iterable[str] = PREFIX_CANDIDATES) -> Dict[str, torch.Tensor]:
    """Remove a shared leading prefix (e.g. 'module.') when present on all keys."""
    result = dict(state)
    changed = True
    while changed and result:
        changed = False
        keys = list(result.keys())
        for prefix in prefixes:
            if all(k.startswith(prefix) for k in keys):
                result = {k[len(prefix) :]: v for k, v in result.items()}
                changed = True
                break
    return result


def _is_seg_or_head(key: str) -> bool:
    return (
        ".seg_layers." in key
        or key.endswith(".seg_layers")
        or "final_conv" in key
        or "output_block" in key
    )


class nnUNetTrainerMultiEncodernnUNetText(nnUNetTrainerMultiEncodernnUNet):
    """Fusion-style multi-encoder trainer with text conditioning and aux losses."""

    def initialize(self):
        # Call base nnUNetTrainer.initialize directly to avoid legacy checkpoint loading in
        # nnUNetTrainerMultiEncodernnUNet.initialize (we only want encoder pretraining handled below).
        nnUNetTrainer.initialize(self)
        try:
            self._maybe_load_encoder_pretrain()
        except Exception as exc:  # pragma: no cover - defensive: we must not crash training
            self.print_to_log_file(f"[Encoder-Pretrain] skipped due to error: {exc}")
        # Optional runtime overrides (batch size, iterations per epoch)
        try:
            bs = os.environ.get('NNUNET_BATCH_SIZE', '').strip()
            if bs:
                self.batch_size = int(bs)
                self.print_to_log_file(f"Overriding batch_size via env: {self.batch_size}")
            iters = os.environ.get('NNUNET_ITERS_PER_EPOCH', '').strip()
            if iters:
                self.num_iterations_per_epoch = int(iters)
                self.print_to_log_file(f"Overriding num_iterations_per_epoch via env: {self.num_iterations_per_epoch}")
        except Exception:
            pass

        self._prepare_text_embedding()
        self._ensure_text_for_modulation()
        try:
            net = self.network.module if hasattr(self.network, 'module') else self.network
            self.print_to_log_file(
                f"[text_head] use_alignment_head={getattr(net, 'use_alignment_head', None)} "
                f"return_heatmap={getattr(net, 'return_heatmap', None)} "
                f"text_embed_dim={getattr(net, 'text_embed_dim', None)}"
            )
        except Exception:
            pass
        self.lambda_align = float(os.environ.get('NNUNET_LAMBDA_ALIGN', '0.0'))
        self.lambda_heat = float(os.environ.get('NNUNET_LAMBDA_HEAT', '0.0'))
        try:
            self.aux_warmup_epochs = int(os.environ.get('NNUNET_AUX_WARMUP_EPOCHS', '0'))
            self.aux_ramp_epochs = int(os.environ.get('NNUNET_AUX_RAMP_EPOCHS', '0'))
        except Exception:
            self.aux_warmup_epochs, self.aux_ramp_epochs = 0, 0
        self._aux_scale = 1.0 if (self.aux_warmup_epochs == 0 and self.aux_ramp_epochs == 0) else 0.0

        self.save_heatmaps = os.environ.get('NNUNET_SAVE_HEATMAPS', '0') in ('1', 'true', 'True')
        self.export_full_heatmap = os.environ.get('NNUNET_EXPORT_FULL_HEATMAP', '0') in ('1', 'true', 'True')
        self.heatmap_dirname = os.environ.get('NNUNET_HEATMAP_DIRNAME', 'validation_heatmaps_npz')
        self.export_heatmap_nifti = os.environ.get('NNUNET_EXPORT_HEATMAP_NIFTI', '0') in ('1', 'true', 'True')

        try:
            if 'NNUNET_OVERSAMPLE_FG' in os.environ:
                self.oversample_foreground_percent = float(os.environ['NNUNET_OVERSAMPLE_FG'])
            if 'NNUNET_VAL_ITERS' in os.environ:
                self.num_val_iterations_per_epoch = int(os.environ['NNUNET_VAL_ITERS'])
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Text embedding helpers
    # ------------------------------------------------------------------

    def _ensure_text_for_modulation(self):
        """If decoder expects a text embedding make sure at least a learnable vector exists."""
        try:
            decoder = getattr(self.network, 'decoder', None)
            if decoder is None:
                return
            uses_modulation = any(not isinstance(m, nn.Identity) for m in getattr(decoder, 'modulators', []))
            align_needed = bool(getattr(self.network, 'use_alignment_head', False))
            dim = getattr(decoder, 'text_embed_dim', None)
            if (uses_modulation or align_needed) and self.text_embed is None and dim is not None:
                self.text_embed_param = nn.Parameter(torch.zeros(1, int(dim), device=self.device))
                self.optimizer.add_param_group({'params': [self.text_embed_param]})
        except Exception:
            pass

    def _prepare_text_embedding(self):
        self.text_embed = None
        dim_env = os.environ.get('NNUNET_TEXT_EMBED_DIM')
        if not dim_env:
            return
        dim = int(dim_env)

        def _truthy(val: str) -> bool:
            return str(val).lower() in ('1', 'true', 'yes', 'y')

        skip_openclip = _truthy(os.environ.get('NNUNET_SKIP_OPENCLIP', '0')) or _truthy(os.environ.get('HF_HUB_OFFLINE', '0'))
        path = os.environ.get('NNUNET_TEXT_EMBED_PATH')
        if path and os.path.isfile(path):
            vec = np.load(path)
            vec = vec.astype(np.float32)
            if vec.ndim == 1:
                vec = vec[None]
            self.text_embed = torch.from_numpy(vec)
        else:
            prompts: List[str] = []
            prompts_env = os.environ.get('NNUNET_TEXT_PROMPTS')
            if prompts_env:
                prompts = [p.strip() for p in prompts_env.replace('||', ',').split(',') if p.strip()]
            prompts_file = os.environ.get('NNUNET_TEXT_PROMPTS_FILE')
            if prompts_file and os.path.isfile(prompts_file):
                with open(prompts_file, 'r') as fh:
                    for line in fh:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            prompts.append(line)
            if not prompts:
                single = os.environ.get('NNUNET_TEXT_PROMPT')
                if single:
                    prompts = [single]

            if prompts and not skip_openclip:
                try:
                    from open_clip import create_model_from_pretrained, get_tokenizer

                    model_id = os.environ.get(
                        'NNUNET_TEXT_MODEL',
                        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
                    )
                    model, _ = create_model_from_pretrained(model_id)
                    model = model.eval().to(self.device)
                    tokenizer = get_tokenizer(model_id)
                    tokens = tokenizer(prompts, context_length=256).to(self.device)
                    with torch.no_grad():
                        embed = model.encode_text(tokens).float()
                        embed = embed / (embed.norm(dim=-1, keepdim=True) + 1e-6)
                        embed = embed.mean(dim=0, keepdim=True)
                        embed = embed / (embed.norm(dim=-1, keepdim=True) + 1e-6)
                    self.text_embed = embed.detach().cpu()
                    self.print_to_log_file(f"Loaded BiomedCLIP text embedding from {len(prompts)} prompts")
                except Exception as exc:
                    self.print_to_log_file(f"BiomedCLIP init failed ({exc}); continuing without fixed text embedding")
                    self.text_embed = None

        if self.text_embed is not None:
            if self.text_embed.shape[-1] != dim:
                self.print_to_log_file(
                    f"Projecting text embedding from dim {self.text_embed.shape[-1]} to requested {dim}")
                proj = torch.empty(self.text_embed.shape[-1], dim)
                torch.manual_seed(0)
                nn.init.orthogonal_(proj)
                self.text_embed = self.text_embed @ proj
            self.text_embed = self.text_embed.to(self.device)

    # ------------------------------------------------------------------
    # Pretrain loading
    # ------------------------------------------------------------------

    def _maybe_load_encoder_pretrain(self):
        ckpt_path = os.environ.get('NNUNET_ENCODER_PRETRAIN', '').strip()
        if not ckpt_path:
            return
        if not os.path.isfile(ckpt_path):
            self.print_to_log_file(f"[Encoder-Pretrain] missing checkpoint: {ckpt_path}")
            return
        try:
            try:
                saved = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            except TypeError:
                saved = torch.load(ckpt_path, map_location='cpu')
        except Exception as exc:
            self.print_to_log_file(f"[Encoder-Pretrain] failed to load checkpoint: {exc}")
            return

        if isinstance(saved, dict):
            state = None
            for key in ('network_weights', 'state_dict', 'model_state', 'model_state_dict'):
                val = saved.get(key)
                if isinstance(val, dict):
                    state = val
                    break
            if state is None:
                state = {k: v for k, v in saved.items() if isinstance(v, torch.Tensor)}
        else:
            state = saved

        if not isinstance(state, dict) or not state:
            self.print_to_log_file(
                f"[Encoder-Pretrain] checkpoint {ckpt_path} does not contain tensor weights; aborting")
            return

        tensors = {k: v for k, v in state.items() if isinstance(v, torch.Tensor)}
        if not tensors:
            self.print_to_log_file(f"[Encoder-Pretrain] no tensor parameters found in {ckpt_path}")
            return

        tensors = _strip_common_prefix(tensors)
        net = self.network.module if isinstance(self.network, DDP) else self.network
        target_sd = net.state_dict()
        encoder_count = len(getattr(net, 'encoders', [])) or 1

        mapped = self._map_state_to_encoders(tensors, target_sd, encoder_count)
        if mapped:
            mapped = {k: v for k, v in mapped.items() if k.startswith('encoders.')}
        if not mapped:
            self.print_to_log_file(f"[Encoder-Pretrain] found no compatible encoder weights in {ckpt_path}")
            return

        result = net.load_state_dict(mapped, strict=False)
        miss = getattr(result, 'missing_keys', [])
        unexp = getattr(result, 'unexpected_keys', [])
        self.print_to_log_file(
            f"[Encoder-Pretrain] loaded={len(mapped)} missing={len(miss)} unexpected={len(unexp)} from={ckpt_path}")
        if miss:
            self.print_to_log_file(f"[Encoder-Pretrain] example missing key: {miss[:5]}")
        if unexp:
            self.print_to_log_file(f"[Encoder-Pretrain] example unexpected key: {unexp[:5]}")

    def _map_state_to_encoders(
        self,
        state: Dict[str, torch.Tensor],
        target_sd: Dict[str, torch.Tensor],
        encoder_count: int,
    ) -> Dict[str, torch.Tensor]:
        mapped: Dict[str, torch.Tensor] = {}

        def assign(target_key: str, tensor: torch.Tensor) -> bool:
            tgt = target_sd.get(target_key)
            if tgt is None:
                return False
            if tensor.shape != tgt.shape:
                return False
            mapped[target_key] = tensor.clone()
            return True

        def adapt_first_conv(src_key: str, tensor: torch.Tensor, target_key: str) -> torch.Tensor | None:
            tgt = target_sd.get(target_key)
            if tgt is None:
                return None
            if tensor.shape == tgt.shape:
                return tensor
            if (
                tensor.ndim == tgt.ndim
                and tensor.shape[0] == tgt.shape[0]
                and tensor.shape[2:] == tgt.shape[2:]
                and tgt.shape[1] == 1
                and tensor.shape[1] != 1
            ):
                return tensor.mean(dim=1, keepdim=True)
            if (
                tensor.ndim == tgt.ndim
                and tensor.shape[0] == tgt.shape[0]
                and tensor.shape[1] == tgt.shape[1]
                and tensor.shape[3:] == tgt.shape[3:]
                and tensor.shape[2] == 1
                and tgt.shape[2] > 1
            ):
                reps = [1] * tensor.ndim
                reps[2] = tgt.shape[2]
                return tensor.repeat(tuple(reps)) / float(reps[2])
            if (
                tensor.ndim == tgt.ndim
                and tensor.shape[0] == tgt.shape[0]
                and tensor.shape[1] == tgt.shape[1]
                and tensor.shape[3:] == tgt.shape[3:]
                and tgt.shape[2] == 1
            ):
                return tensor.mean(dim=2, keepdim=True)
            self.print_to_log_file(
                f"[Encoder-Pretrain] shape mismatch for {src_key}->{target_key}: {tuple(tensor.shape)} vs {tuple(tgt.shape)}")
            return None

        # Case A: checkpoint already stores encoders.i.*
        enc_groups: Dict[int, List[Tuple[str, torch.Tensor]]] = {}
        for key, tensor in state.items():
            if not key.startswith('encoders.') or _is_seg_or_head(key):
                continue
            parts = key.split('.')
            if len(parts) < 3:
                continue
            try:
                idx = int(parts[1])
            except ValueError:
                continue
            enc_groups.setdefault(idx, []).append((key, tensor))

        if enc_groups:
            for idx in range(encoder_count):
                source_idx = idx if idx in enc_groups else 0 if 0 in enc_groups else sorted(enc_groups.keys())[0]
                for key, tensor in enc_groups[source_idx]:
                    if source_idx == idx:
                        assign(key, tensor)
                    else:
                        target_key = key.replace(f'encoders.{source_idx}.', f'encoders.{idx}.', 1)
                        assign(target_key, tensor)
            return mapped

        # Case B: plain nnUNet encoder.* layout
        plain_items = [(k, v) for k, v in state.items() if k.startswith('encoder.') and not _is_seg_or_head(k)]
        if plain_items:
            for idx in range(encoder_count):
                for key, tensor in plain_items:
                    tail = key[len('encoder.') :]
                    target_key = f'encoders.{idx}.{tail}'
                    adapted = adapt_first_conv(key, tensor, target_key)
                    if adapted is not None:
                        assign(target_key, adapted)
            return mapped

        # Case C: legacy encoder.stages.* (original nnUNet)
        legacy = OrderedDict(
            (k, v)
            for k, v in state.items()
            if k.startswith('encoder.stages.') and not _is_seg_or_head(k)
        )
        if legacy:
            for key, tensor in legacy.items():
                parts = key.split('.')
                if len(parts) < 7 or parts[4] != 'convs' or parts[6] == 'all_modules':
                    continue
                stage_idx = parts[2]
                conv_idx = parts[5]
                attr = '.'.join(parts[6:])
                tail = f'stages.{stage_idx}.convs.{conv_idx}.{attr}'
                for idx in range(encoder_count):
                    target_key = f'encoders.{idx}.{tail}'
                    adapted = adapt_first_conv(key, tensor, target_key)
                    if adapted is not None:
                        assign(target_key, adapted)
            return mapped

        # Case D: last resort shape matching against encoder 0
        enc0_keys = [k for k in target_sd.keys() if k.startswith('encoders.0.') and not _is_seg_or_head(k)]
        for target_key in enc0_keys:
            tgt = target_sd[target_key]
            for src_key, tensor in state.items():
                if tensor.shape == tgt.shape:
                    if assign(target_key, tensor):
                        break
        if mapped:
            # replicate to all encoders
            for idx in range(1, encoder_count):
                for key, tensor in list(mapped.items()):
                    if not key.startswith('encoders.0.'):
                        continue
                    target_key = key.replace('encoders.0.', f'encoders.{idx}.', 1)
                    assign(target_key, tensor)
        return mapped

    # ------------------------------------------------------------------
    # Loss / training loop overrides
    # ------------------------------------------------------------------

    def _build_loss(self):
        from nnunetv2.training.loss.compound_losses import DC_and_topk_loss
        from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
        from nnunetv2.training.loss.robust_ce_loss import TopKLoss
        from nnunetv2.training.loss.tversky import FocalTverskyLoss, SoftTverskyLoss

        assert not self.label_manager.has_regions, "regions not supported by this trainer"

        kind = os.environ.get('NNUNET_TEXT_LOSS', 'dice_topk').strip().lower()
        ignore_label = self.label_manager.ignore_label

        if kind in ('dice_topk', 'dice', 'dc_topk'):
            loss: nn.Module = DC_and_topk_loss(
                {"batch_dice": True, "smooth": 1e-5, "do_bg": False, "ddp": self.is_ddp},
                {"k": 10, "label_smoothing": 0.05},
                weight_ce=1,
                weight_dice=1,
                ignore_label=ignore_label,
            )
        else:
            ce = TopKLoss(k=10, label_smoothing=0.05)
            try:
                alpha = float(os.environ.get('NNUNET_TVERSKY_ALPHA', '0.3'))
                beta = float(os.environ.get('NNUNET_TVERSKY_BETA', '0.7'))
                gamma = float(os.environ.get('NNUNET_FTVERSKY_GAMMA', '1.5'))
            except Exception:
                alpha, beta, gamma = 0.3, 0.7, 1.5

            if kind in ('tversky_topk', 'tversky'):
                tv = SoftTverskyLoss(batch_dice=True, do_bg=False, ddp=self.is_ddp, alpha=alpha, beta=beta)
            elif kind in ('focal_tversky_topk', 'focal_tversky', 'ftversky_topk', 'ftversky'):
                tv = FocalTverskyLoss(batch_dice=True, do_bg=False, ddp=self.is_ddp, alpha=alpha, beta=beta, gamma=gamma)
            else:
                tv = SoftTverskyLoss(batch_dice=True, do_bg=False, ddp=self.is_ddp, alpha=alpha, beta=beta)

            class _TVerskyTopKLoss(nn.Module):
                def __init__(self, tv_loss: nn.Module, ce_loss: nn.Module, ignore_label_val):
                    super().__init__()
                    self.tv = tv_loss
                    self.ce = ce_loss
                    self.ignore_label = ignore_label_val

                def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                    if self.ignore_label is not None:
                        mask = (target != self.ignore_label).bool()
                        target_tv = torch.clone(target)
                        target_tv[target == self.ignore_label] = 0
                    else:
                        mask = None
                        target_tv = target
                    tv_loss = self.tv(net_output, target_tv, loss_mask=mask)
                    if self.ignore_label is not None and (mask is None or mask.sum() == 0):
                        ce_loss = 0.0
                    else:
                        ce_loss = self.ce(net_output, target)
                    return tv_loss + ce_loss

            loss = _TVerskyTopKLoss(tv, ce, ignore_label)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

    def _compute_aux_losses(self, extras: dict, target: Union[List[torch.Tensor], torch.Tensor]) -> float:
        if extras is None or (self.lambda_align == 0 and self.lambda_heat == 0):
            return 0.0

        if isinstance(target, list):
            gt = target[0]
        else:
            gt = target

        sim = extras.get('sim') if isinstance(extras, dict) else None
        if sim is None:
            return self.lambda_align * 0.0

        with torch.no_grad():
            if gt.shape[1] > 1:
                fg_mask = (gt[:, 1:, ...].sum(1, keepdim=True) > 0).float()
            else:
                fg_mask = (gt > 0).float()

        pos = (sim * fg_mask).sum() / (fg_mask.sum() + 1e-6)
        align_loss = 1.0 - pos

        if sim.dtype != torch.float32:
            sim = sim.float()
        if fg_mask.dtype != torch.float32:
            fg_mask = fg_mask.float()
        sim = sim.clamp(1e-6, 1 - 1e-6)
        if not torch.isfinite(sim).all():
            return float(self.lambda_align) * align_loss
        with torch.autocast(self.device.type, enabled=False) if self.device.type == 'cuda' else dummy_context():
            heat_loss = F.binary_cross_entropy(sim, fg_mask)

        scale = getattr(self, '_aux_scale', 1.0)
        total = float(self.lambda_align * scale) * align_loss + float(self.lambda_heat * scale) * heat_loss

        stats = getattr(self, '_aux_stats', None)
        if stats is None:
            stats = {'align': [], 'heat': [], 'total': []}
            setattr(self, '_aux_stats', stats)
        for key, value in (("align", align_loss), ("heat", heat_loss), ("total", total)):
            try:
                stats[key].append(float(value.detach().cpu().item() if isinstance(value, torch.Tensor) else value))
            except Exception:
                pass

        return total

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(self.device.type, enabled=self.use_amp) if self.device.type == 'cuda' else dummy_context():
            text = self.text_embed
            if text is not None:
                if text.shape[0] == 1 and data.shape[0] > 1:
                    text = text.expand(data.shape[0], -1)
                else:
                    text = text[: data.shape[0]]
            elif hasattr(self, 'text_embed_param'):
                text = self.text_embed_param[:1].expand(data.shape[0], -1)

            try:
                output = self.network(data, text=text, return_extra=True)
            except TypeError:
                output = self.network(data)
                text = None

            if isinstance(output, dict):
                seg = output.get('seg')
                extras = {k: v for k, v in output.items() if k != 'seg'}
            else:
                seg, extras = output, None

            loss = self.loss(seg, target)
            if text is not None and extras is not None:
                loss = loss + self._compute_aux_losses(extras, target)

        if not torch.isfinite(loss):
            self.print_to_log_file('WARNING: non-finite loss detected; skipping optimizer step')
            self.optimizer.zero_grad(set_to_none=True)
            return {'loss': float(0.0)}

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': float(loss.detach().cpu().item())}

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        # reset per-epoch auxiliary statistics
        try:
            if hasattr(self, '_aux_stats'):
                for k in self._aux_stats:
                    self._aux_stats[k].clear()
        except Exception:
            self._aux_stats = {'align': [], 'heat': [], 'total': []}
        try:
            warmup = int(getattr(self, 'aux_warmup_epochs', 0) or 0)
            ramp = int(getattr(self, 'aux_ramp_epochs', 0) or 0)
        except Exception:
            warmup, ramp = 0, 0
        epoch = int(getattr(self, 'current_epoch', 0) or 0)

        def _baseline_scale(ep: int) -> float:
            if warmup <= 0 and ramp <= 0:
                return 1.0
            if ep < warmup:
                return 0.0
            if ramp > 0 and ep < warmup + ramp:
                return max(0.0, min(1.0, (ep - warmup + 1) / float(ramp)))
            return 1.0

        scale = _baseline_scale(epoch)

        def _truthy(val: str) -> bool:
            return str(val).lower() in ('1', 'true', 'yes', 'y')

        dyn_enable = _truthy(os.environ.get('NNUNET_AUX_DYNAMIC', '1'))
        try:
            k = int(os.environ.get('NNUNET_AUX_LOSS_STABLE_EPOCHS', '5'))
        except Exception:
            k = 5
        try:
            rel_thr = float(os.environ.get('NNUNET_AUX_LOSS_REL_CHANGE', '0.05'))
        except Exception:
            rel_thr = 0.05
        try:
            ema_thr = float(os.environ.get('NNUNET_AUX_EMA_DICE_THRESH', '0.30'))
        except Exception:
            ema_thr = 0.30
        try:
            hard = os.environ.get('NNUNET_AUX_HARD_START_EPOCH', '').strip()
            hard_start = int(hard) if hard else (warmup + ramp if (warmup + ramp) > 0 else warmup)
        except Exception:
            hard_start = warmup + ramp if (warmup + ramp) > 0 else warmup

        if dyn_enable and ramp > 0 and epoch >= warmup:
            stable = False
            try:
                logs = getattr(self, 'logger', None)
                hist = getattr(logs, 'my_fantastic_logging', None)
                if hist:
                    train_losses = hist.get('train_losses', [])
                    ema_fg = hist.get('ema_fg_dice', [])
                    if len(train_losses) >= 2 * k and k > 0:
                        m2 = float(np.mean(train_losses[-k:]))
                        m1 = float(np.mean(train_losses[-2 * k:-k]))
                        denom = max(1e-8, abs(m1))
                        rel_change = abs(m2 - m1) / denom
                        loss_stable = rel_change < rel_thr
                    else:
                        loss_stable = False
                    ema_ok = bool(ema_fg) and float(ema_fg[-1]) >= ema_thr
                    stable = loss_stable or ema_ok
            except Exception:
                stable = False

            if not stable and epoch < hard_start:
                scale = 0.0
            else:
                scale = _baseline_scale(epoch)

        self._aux_scale = float(scale)
        try:
            self.print_to_log_file(
                f"[aux_schedule] epoch={epoch} warmup={warmup} ramp={ramp} hard_start={hard_start} dyn={int(dyn_enable)} scale={self._aux_scale:.3f}")
        except Exception:
            pass

    def on_train_epoch_end(self, train_outputs=None):
        super().on_train_epoch_end(train_outputs)
        stats = getattr(self, '_aux_stats', None)
        if not stats:
            return
        try:
            summary = {k: float(np.mean(v)) if len(v) > 0 else float('nan') for k, v in stats.items()}
            self.print_to_log_file(
                f"[aux_summary] epoch={getattr(self, 'current_epoch', 'NA')} "
                f"align={summary.get('align', float('nan')):.4f} "
                f"heat={summary.get('heat', float('nan')):.4f} "
                f"total={summary.get('total', float('nan')):.4f}"
            )
        except Exception:
            pass
