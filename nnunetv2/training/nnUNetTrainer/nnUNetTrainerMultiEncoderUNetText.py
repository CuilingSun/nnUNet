import os
from typing import List, Optional, Union

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, join
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.inference.sliding_window_prediction import compute_gaussian, compute_steps_for_sliding_window
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMultiEncoderUNet import (
    nnUNetTrainerMultiEncoderUNet,
)


class nnUNetTrainerMultiEncoderUNetText(nnUNetTrainerMultiEncoderUNet):
    """
    Extends the MultiEncoderUNet trainer with optional text-conditioning losses.
    - Reads feature-dimension & behavior from environment variables (non-breaking defaults)
    - If text is unavailable, falls back to vanilla nnU-Net training.

    Env options (all optional):
      NNUNET_TEXT_EMBED_DIM: int
      NNUNET_TEXT_EMBED_PATH: path to .npy vector
      NNUNET_TEXT_PROMPT: single prompt string (deprecated if PROMPTS is set)
      NNUNET_TEXT_PROMPTS: multiple prompts separated by '||' or commas
      NNUNET_TEXT_PROMPTS_FILE: text file with one prompt per line
      NNUNET_TEXT_MODEL: open_clip model id (default BiomedCLIP base)
      NNUNET_TEXT_MODULATION: 'none' | 'film' | 'gate'
      NNUNET_USE_ALIGNMENT_HEAD: '0' | '1'
      NNUNET_RETURN_HEATMAP: '0' | '1'
      NNUNET_HEATMAP_T: float
      NNUNET_LAMBDA_ALIGN: float (weight for L_align)
      NNUNET_LAMBDA_HEAT: float (weight for L_heat)
    """

    def initialize(self):
        super().initialize()
        # Optionally load a pretrained encoder (image branch only) from a checkpoint.
        # Triggered by env var NNUNET_ENCODER_PRETRAIN pointing to a .pth file.
        # Supports:
        #  - Same-arch MultiEncoderUNet checkpoints (encoders.0.* keys)
        #  - Plain single-encoder nnUNet-style checkpoints with 'encoder.*' keys
        #  - Safe shape filtering with first-conv in_channels adaptation (e.g., >1 -> 1 via mean)
        try:
            self._maybe_load_image_encoder_pretrain()
        except Exception as e:
            self.print_to_log_file(f"[WARN] Encoder pretrain loading skipped due to error: {e}")
        # Optional runtime overrides to ease debugging/perf tuning
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
        # Prepare a fixed or learnable text embedding (if required)
        self._prepare_text_embedding()
        self._ensure_text_for_modulation()
        self.lambda_align = float(os.environ.get('NNUNET_LAMBDA_ALIGN', '0.0'))
        self.lambda_heat = float(os.environ.get('NNUNET_LAMBDA_HEAT', '0.0'))
        # Optional epoch-based curriculum for auxiliary (heatmap/alignment) losses
        try:
            self.aux_warmup_epochs = int(os.environ.get('NNUNET_AUX_WARMUP_EPOCHS', '0'))
            self.aux_ramp_epochs = int(os.environ.get('NNUNET_AUX_RAMP_EPOCHS', '0'))
        except Exception:
            self.aux_warmup_epochs, self.aux_ramp_epochs = 0, 0
        self._aux_scale = 1.0 if (self.aux_warmup_epochs == 0 and self.aux_ramp_epochs == 0) else 0.0
        self.save_heatmaps = os.environ.get('NNUNET_SAVE_HEATMAPS', '0') in ('1', 'true', 'True')

        # export whole-case heatmap control
        self.export_full_heatmap = os.environ.get('NNUNET_EXPORT_FULL_HEATMAP', '0') in ('1', 'true', 'True')
        self.heatmap_dirname = os.environ.get('NNUNET_HEATMAP_DIRNAME', 'validation_heatmaps_npz')
        self.export_heatmap_nifti = os.environ.get('NNUNET_EXPORT_HEATMAP_NIFTI', '0') in ('1', 'true', 'True')
        # Allow quick env-based tuning similar to the non-text trainer
        try:
            if 'NNUNET_OVERSAMPLE_FG' in os.environ:
                self.oversample_foreground_percent = float(os.environ['NNUNET_OVERSAMPLE_FG'])
            if 'NNUNET_VAL_ITERS' in os.environ:
                self.num_val_iterations_per_epoch = int(os.environ['NNUNET_VAL_ITERS'])
        except Exception:
            pass
        self._global_iteration = 0
        self._epoch_iteration = 0
        self._aux_epoch_stats = []
        self._last_aux_stats = None

    def _ensure_text_for_modulation(self):
        """
        Ensure a text embedding is available when required:
        - if decoder uses text modulation (FiLM/Gate)
        - or if alignment head/heatmap is enabled in the network
        In absence of provided embedding, create a learnable parameter.
        """
        try:
            dec = getattr(self.network, 'decoder', None)
            if dec is None:
                return
            uses_modulation = any(not isinstance(m, nn.Identity) for m in getattr(dec, 'modulators', []))
            # also consider alignment head on the main network
            align_needed = bool(getattr(self.network, 'use_alignment_head', False))
            dim = getattr(dec, 'text_embed_dim', None)
            if (uses_modulation or align_needed) and self.text_embed is None and dim is not None:
                # create a learnable parameter initialized to zeros
                self.text_embed_param = nn.Parameter(torch.zeros(1, int(dim), device=self.device))
                # add to optimizer so it learns along with the network
                self.optimizer.add_param_group({'params': [self.text_embed_param]})
        except Exception:
            # silently skip if anything unexpected happens
            pass

    

    def _maybe_load_image_encoder_pretrain(self):
        """
        Load pretrained weights for the image encoder only from a given checkpoint.
        Activate by setting env NNUNET_ENCODER_PRETRAIN to a .pth path.

        Supports two common layouts:
          - MultiEncoderUNet/Multi-branch: keys like 'encoders.0.*' (direct filter)
          - Plain single-encoder nnUNet:  keys like 'encoder.*' mapped to 'encoders.0.*'

        Safety:
          - Skip segmentation heads and unrelated modules
          - Filter by exact shape; adapt first conv in_channels: N->1 via mean across channel dim
          - Strict=False load so non-matching keys are ignored
        """
        import os
        import torch
        from torch.nn.parallel import DistributedDataParallel as DDP

        ckpt_path = os.environ.get('NNUNET_ENCODER_PRETRAIN', '').strip()
        if not ckpt_path or not os.path.isfile(ckpt_path):
            return

        try:
            try:
                saved = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            except TypeError:
                # PyTorch <2.4 doesn't support weights_only kwarg
                saved = torch.load(ckpt_path, map_location='cpu')
        except Exception as e:
            self.print_to_log_file(f"[Encoder-Pretrain] Failed to load checkpoint: {e}")
            return

        state = saved.get('network_weights') or saved.get('state_dict') or saved
        net = self.network.module if isinstance(self.network, DDP) else self.network
        target_sd = net.state_dict()

        def _is_seg_or_head(k: str) -> bool:
            return ('.seg_layers.' in k) or k.endswith('.seg_layers') or ('final_conv' in k) or ('output_block' in k)

        src_keys = list(state.keys())
        has_multi = any(k.startswith('encoders.0.') for k in src_keys)
        has_plain = any(k.startswith('encoder.') for k in src_keys)

        mapped = {}
        if has_multi:
            for k, v in state.items():
                if not k.startswith('encoders.0.') or _is_seg_or_head(k):
                    continue
                if k in target_sd and target_sd[k].shape == v.shape:
                    mapped[k] = v
        elif has_plain:
            for k, v in state.items():
                if not k.startswith('encoder.') or _is_seg_or_head(k):
                    continue
                tk = 'encoders.0.' + k[len('encoder.'):]
                if tk in target_sd:
                    tv = v
                    # first conv adaptation: collapse in_channels to 1 via mean if needed
                    if (
                        isinstance(tv, torch.Tensor)
                        and isinstance(target_sd[tk], torch.Tensor)
                        and tv.ndim >= 3 and target_sd[tk].ndim == tv.ndim
                        and tv.shape[0] == target_sd[tk].shape[0]
                        and tv.shape[2:] == target_sd[tk].shape[2:]
                        and target_sd[tk].shape[1] == 1 and tv.shape[1] != 1
                    ):
                        tv = tv.mean(dim=1, keepdim=True)
                    if target_sd[tk].shape == tv.shape:
                        mapped[tk] = tv
        else:
            # Fallback: try best-effort shape-based matching for encoders.0.* keys
            for tk, tv in target_sd.items():
                if not tk.startswith('encoders.0.') or _is_seg_or_head(tk):
                    continue
                if tk in state and isinstance(state[tk], torch.Tensor) and state[tk].shape == tv.shape:
                    mapped[tk] = state[tk]
                    continue
                # Single pass: find a same-shape tensor not obviously a head
                for sk, sv in state.items():
                    if _is_seg_or_head(sk):
                        continue
                    if isinstance(sv, torch.Tensor) and sv.shape == tv.shape:
                        mapped[tk] = sv
                        break

        # Optional: copy encoder 0 weights to all encoders if shapes match
        copy_all = str(os.environ.get('NNUNET_COPY_PRETRAIN_TO_ALL_ENCODERS', '0')).lower() in ('1', 'true', 'yes', 'y')
        if copy_all:
            try:
                num_enc = len(getattr(net, 'encoders', []))
            except Exception:
                num_enc = 1
            if num_enc > 1:
                extra = {}
                for i in range(1, num_enc):
                    for k, v in mapped.items():
                        if not k.startswith('encoders.0.'):
                            continue
                        kk = k.replace('encoders.0.', f'encoders.{i}.', 1)
                        if kk in target_sd and target_sd[kk].shape == v.shape:
                            extra[kk] = v
                mapped.update(extra)

        res = net.load_state_dict(mapped, strict=False)
        try:
            miss_ct = len(getattr(res, 'missing_keys', []))
            unexp_ct = len(getattr(res, 'unexpected_keys', []))
        except Exception:
            miss_ct = unexp_ct = 0
        self.print_to_log_file(f"[Encoder-Pretrain] from={ckpt_path}")
        self.print_to_log_file(f"[Encoder-Pretrain] loaded_params={len(mapped)} missing={miss_ct} unexpected={unexp_ct}")

    def _build_loss(self):
        """Build primary segmentation loss with optional Text-variant selection.

        Supported via env NNUNET_TEXT_LOSS (case-insensitive):
          - 'dice_topk' (default)
          - 'tversky_topk' (use SoftTverskyLoss + TopK CE)
          - 'focal_tversky_topk' (use FocalTverskyLoss + TopK CE)
        Tversky parameters can be configured via NNUNET_TVERSKY_ALPHA, NNUNET_TVERSKY_BETA, NNUNET_FTVERSKY_GAMMA.
        """
        from nnunetv2.training.loss.compound_losses import DC_and_topk_loss
        from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
        from nnunetv2.training.loss.tversky import SoftTverskyLoss, FocalTverskyLoss
        from nnunetv2.training.loss.robust_ce_loss import TopKLoss
        import numpy as np
        import os
        import torch
        import torch.nn as nn

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
            # Build CE (TopK) component shared by Tversky variants
            ce = TopKLoss(k=10, label_smoothing=0.05)
            # Build Tversky base
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
                # fallback to default
                tv = SoftTverskyLoss(batch_dice=True, do_bg=False, ddp=self.is_ddp, alpha=alpha, beta=beta)

            class _TVerskyTopKCE(nn.Module):
                def __init__(self, tv_loss: nn.Module, ce_loss: nn.Module, ignore_label_val):
                    super().__init__()
                    self.tv = tv_loss
                    self.ce = ce_loss
                    self.ignore_label = ignore_label_val

                def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                    if self.ignore_label is not None:
                        # mask out ignore label for Tversky
                        mask = (target != self.ignore_label).bool()
                        target_tv = torch.clone(target)
                        target_tv[target == self.ignore_label] = 0
                        num_fg = mask.sum()
                    else:
                        target_tv = target
                        mask = None
                        num_fg = None
                    tv_loss = self.tv(net_output, target_tv, loss_mask=mask)
                    ce_loss = self.ce(net_output, target) if (self.ignore_label is None or (num_fg is not None and num_fg > 0)) else 0.0
                    return tv_loss + ce_loss

            loss = _TVerskyTopKCE(tv, ce, ignore_label)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

    def _prepare_text_embedding(self):
        self.text_embed = None
        dim_env = os.environ.get('NNUNET_TEXT_EMBED_DIM', None)
        if dim_env is None:
            return
        dim = int(dim_env)
        # Allow skipping OpenCLIP when offline or explicitly requested
        def _truthy(x: str) -> bool:
            return str(x).lower() in ('1', 'true', 'yes', 'y')
        skip_openclip = _truthy(os.environ.get('NNUNET_SKIP_OPENCLIP', '0')) or \
                         _truthy(os.environ.get('HF_HUB_OFFLINE', '0'))
        # 1) load from file if available
        p = os.environ.get('NNUNET_TEXT_EMBED_PATH', None)
        if p and os.path.isfile(p):
            vec = np.load(p)
            vec = vec.astype(np.float32)
            if vec.ndim == 1:
                vec = vec[None]
            self.text_embed = torch.from_numpy(vec)
        else:
            # 2) try prompts via open_clip / BiomedCLIP
            #    support multiple prompts provided via env or file
            prompts = []
            prompts_env = os.environ.get('NNUNET_TEXT_PROMPTS', None)
            if prompts_env:
                # split by '||' or comma
                if '||' in prompts_env:
                    prompts = [s.strip() for s in prompts_env.split('||') if s.strip()]
                else:
                    prompts = [s.strip() for s in prompts_env.split(',') if s.strip()]
            file_env = os.environ.get('NNUNET_TEXT_PROMPTS_FILE', None)
            if file_env and os.path.isfile(file_env):
                with open(file_env, 'r') as f:
                    lines = [l.strip() for l in f.readlines()]
                    for l in lines:
                        if l and not l.startswith('#'):
                            prompts.append(l)
            # fallback to single prompt if provided
            single_prompt = os.environ.get('NNUNET_TEXT_PROMPT', None)
            if single_prompt and not prompts:
                prompts = [single_prompt]

            if prompts and not skip_openclip:
                try:
                    from open_clip import create_model_from_pretrained, get_tokenizer
                    model_id = os.environ.get(
                        'NNUNET_TEXT_MODEL',
                        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
                    )
                    model, _ = create_model_from_pretrained(model_id)
                    model = model.eval().to(self.device)
                    tok = get_tokenizer(model_id)
                    tokens = tok(prompts, context_length=256).to(self.device)
                    with torch.no_grad():
                        t = model.encode_text(tokens).float()  # [N, D]
                        t = t / (t.norm(dim=-1, keepdim=True) + 1e-6)
                        # aggregate prompts: mean then re-normalize
                        t = t.mean(dim=0, keepdim=True)
                        t = t / (t.norm(dim=-1, keepdim=True) + 1e-6)
                    self.text_embed = t.detach().cpu()
                    self.print_to_log_file(f"Using BiomedCLIP text embedding from {len(prompts)} prompts")
                except Exception as e:
                    self.print_to_log_file(f"BiomedCLIP init failed ({e}). Falling back to no text.")
                    self.text_embed = None
            else:
                self.text_embed = None

        # pad/trim to desired dim if necessary
        if self.text_embed is not None:
            if self.text_embed.shape[-1] != dim:
                self.print_to_log_file(
                    f"WARNING: text dim {self.text_embed.shape[-1]} != NNUNET_TEXT_EMBED_DIM {dim}; projecting.")
                # simple linear proj onto desired dim (cpu) to avoid requiring extra modules
                w = torch.empty(self.text_embed.shape[-1], dim)
                torch.manual_seed(0)
                nn.init.orthogonal_(w)
                self.text_embed = self.text_embed @ w
            self.text_embed = self.text_embed.to(self.device)

    def _update_aux_logging(self, stats: Optional[dict] = None) -> None:
        base_scale = float(getattr(self, '_aux_scale', 1.0))
        defaults = {
            'total': 0.0,
            'align_contrib': 0.0,
            'heat_contrib': 0.0,
            'align_raw': 0.0,
            'heat_raw': 0.0,
            'scale': base_scale,
            'align_weight': float(self.lambda_align),
            'heat_weight': float(self.lambda_heat),
            'align_weight_scaled': float(self.lambda_align * base_scale),
            'heat_weight_scaled': float(self.lambda_heat * base_scale),
        }
        if stats:
            for key, value in stats.items():
                try:
                    defaults[key] = float(value)
                except Exception:
                    defaults[key] = value
        self._last_aux_stats = defaults
        if not hasattr(self, '_aux_epoch_stats') or self._aux_epoch_stats is None:
            self._aux_epoch_stats = []
        self._aux_epoch_stats.append(defaults)

    def _compute_aux_losses(self, extras: dict, target: Union[List[torch.Tensor], torch.Tensor]):
        if extras is None or (self.lambda_align == 0 and self.lambda_heat == 0):
            self._update_aux_logging(None)
            return 0.0

        scale = float(getattr(self, '_aux_scale', 1.0))

        # get highest-resolution seg target if DS
        if isinstance(target, list):
            gt = target[0]
        else:
            gt = target

        # assume 'sim' is in [0,1] after sigmoid and possibly upsampled to match seg size
        sim = extras.get('sim', None)
        if sim is None:
            self._update_aux_logging(None)
            return 0.0

        # L_align: 1 - mean(sim) over positive pixels (ignore background-only slices safely)
        with torch.no_grad():
            # reduce to foreground mask of any class (>0)
            if gt.shape[1] > 1:  # class channels
                fg_mask = (gt[:, 1:, ...].sum(1, keepdim=True) > 0).float()
            else:
                fg_mask = (gt > 0).float()
        pos = (sim * fg_mask).sum() / (fg_mask.sum() + 1e-6)
        l_align = 1.0 - pos

        # L_heat: BCE between sim and hard GT mask (can be extended to soft GT)
        # BCE is unsafe to autocast; compute in float32 with autocast disabled.
        # Add guards for numerical issues and out-of-range values.
        if sim.dtype != torch.float32:
            sim = sim.float()
        if fg_mask.dtype != torch.float32:
            fg_mask = fg_mask.float()
        # strict clamp and finite check
        sim = sim.clamp(1e-6, 1. - 1e-6)
        if not torch.isfinite(sim).all():
            # skip aux loss if numerically unstable
            align_contrib = self.lambda_align * l_align
            stats = {
                'total': float((align_contrib).detach().cpu().item()),
                'align_contrib': float((align_contrib).detach().cpu().item()),
                'heat_contrib': 0.0,
                'align_raw': float(l_align.detach().cpu().item()),
                'heat_raw': 0.0,
                'scale': scale,
                'align_weight_scaled': self.lambda_align,
                'heat_weight_scaled': 0.0,
            }
            self._update_aux_logging(stats)
            return align_contrib
        with torch.autocast(self.device.type, enabled=False) if self.device.type == 'cuda' else dummy_context():
            bce = nn.functional.binary_cross_entropy(sim, fg_mask)

        # apply curriculum scaling if configured
        la = self.lambda_align * scale
        lh = self.lambda_heat * scale
        align_contrib = la * l_align
        heat_contrib = lh * bce
        total = align_contrib + heat_contrib
        stats = {
            'total': float(total.detach().cpu().item()),
            'align_contrib': float(align_contrib.detach().cpu().item()),
            'heat_contrib': float(heat_contrib.detach().cpu().item()),
            'align_raw': float(l_align.detach().cpu().item()),
            'heat_raw': float(bce.detach().cpu().item()),
            'scale': scale,
            'align_weight_scaled': la,
            'heat_weight_scaled': lh,
        }
        self._update_aux_logging(stats)
        return total

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(self.device.type, enabled=self.use_amp) if self.device.type == 'cuda' else dummy_context():
            # choose text source: fixed/env embedding, else learnable param if present
            text = self.text_embed
            if text is not None:
                # ensure batch alignment: expand singleton to batch size if needed
                if text.shape[0] == 1 and data.shape[0] > 1:
                    text = text.expand(data.shape[0], -1)
                else:
                    text = text[: data.shape[0]]
            elif hasattr(self, 'text_embed_param'):
                text = self.text_embed_param[:1].expand(data.shape[0], -1)
            # ask network for extras when available
            try:
                output = self.network(data, text=text, return_extra=True)
            except TypeError:
                # fallback to default signature
                output = self.network(data)
                text = None

            if isinstance(output, dict):
                seg = output['seg']
                extras = {k: v for k, v in output.items() if k != 'seg'}
            else:
                seg, extras = output, None

            seg_loss = self.loss(seg, target)
            aux_total = seg_loss.new_tensor(0.0) if isinstance(seg_loss, torch.Tensor) else 0.0
            aux_logged = False
            if text is not None and extras is not None:
                aux_total = self._compute_aux_losses(extras, target)
                aux_logged = True
            total_loss = seg_loss + aux_total

        if not aux_logged:
            self._update_aux_logging(None)

        # skip non-finite loss to avoid poisoning optimizer state
        if not torch.isfinite(total_loss):
            self.print_to_log_file('WARNING: non-finite loss detected in text trainer; skipping optimizer step')
            self.optimizer.zero_grad(set_to_none=True)
            return {'loss': float(0.0)}

        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        self._global_iteration = getattr(self, '_global_iteration', 0) + 1
        self._epoch_iteration = getattr(self, '_epoch_iteration', 0) + 1
        aux_stats = getattr(self, '_last_aux_stats', None)
        result = {'loss': float(total_loss.detach().cpu().item())}
        if aux_stats is not None:
            result.update({
                'aux_total': float(aux_stats.get('total', 0.0)),
                'aux_align': float(aux_stats.get('align_contrib', 0.0)),
                'aux_heat': float(aux_stats.get('heat_contrib', 0.0)),
            })
        return result

    def on_train_epoch_end(self, train_outputs: List[dict]):
        super().on_train_epoch_end(train_outputs)
        stats_list = getattr(self, '_aux_epoch_stats', None)
        if not stats_list:
            return
        try:
            total_mean = float(np.mean([s.get('total', 0.0) for s in stats_list]))
            align_mean = float(np.mean([s.get('align_contrib', 0.0) for s in stats_list]))
            heat_mean = float(np.mean([s.get('heat_contrib', 0.0) for s in stats_list]))
            raw_align_mean = float(np.mean([s.get('align_raw', 0.0) for s in stats_list]))
            raw_heat_mean = float(np.mean([s.get('heat_raw', 0.0) for s in stats_list]))
            scale_mean = float(np.mean([s.get('scale', 0.0) for s in stats_list]))
        except Exception:
            return
        self.print_to_log_file(
            f"[aux_epoch_summary] epoch={self.current_epoch} total_mean={total_mean:.6f} "
            f"align_mean={align_mean:.6f} heat_mean={heat_mean:.6f} "
            f"raw_align_mean={raw_align_mean:.6f} raw_heat_mean={raw_heat_mean:.6f} "
            f"scale_mean={scale_mean:.3f}"
        )

    def on_train_epoch_start(self):
        """
        Extend base hook to update auxiliary-loss schedule (_aux_scale) with warmup + linear ramp.
        - Warmup: epochs [0, aux_warmup_epochs) -> scale = 0
        - Ramp:   next aux_ramp_epochs epochs linearly increase to 1
        - After:  scale = 1
        """
        super().on_train_epoch_start()
        self._epoch_iteration = 0
        self._aux_epoch_stats = []
        self._last_aux_stats = None
        # base schedule from static warmup+ramp
        try:
            w = int(getattr(self, 'aux_warmup_epochs', 0) or 0)
            r = int(getattr(self, 'aux_ramp_epochs', 0) or 0)
        except Exception:
            w, r = 0, 0
        e = int(getattr(self, 'current_epoch', 0) or 0)

        # dynamic controls via env (enabled by default)
        def _truthy(x: str) -> bool:
            return str(x).lower() in ('1', 'true', 'yes', 'y')
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
            hard_start_env = os.environ.get('NNUNET_AUX_HARD_START_EPOCH', '').strip()
            hard_start = int(hard_start_env) if hard_start_env else (w + r if (w + r) > 0 else w)
        except Exception:
            hard_start = w + r if (w + r) > 0 else w

        # compute baseline scale without dynamics
        def _baseline_scale(epoch: int) -> float:
            if w <= 0 and r <= 0:
                return 1.0
            if epoch < w:
                return 0.0
            if r > 0 and epoch < (w + r):
                return max(0.0, min(1.0, (epoch - w + 1) / float(r)))
            return 1.0

        scale = _baseline_scale(e)

        # dynamic adjustment: delay ramp until stability (loss change small or EMA dice decent),
        # but start at latest by hard_start
        if dyn_enable and r > 0:
            # check stability metrics from logger (previous epochs)
            stable = False
            try:
                logs = getattr(self, 'logger', None)
                if logs is not None and hasattr(logs, 'my_fantastic_logging'):
                    hist = logs.my_fantastic_logging
                    tr = hist.get('train_losses', [])
                    em = hist.get('ema_fg_dice', [])
                    # loss stability: compare mean of last k vs previous k epochs
                    if len(tr) >= 2 * k and k > 0:
                        m2 = float(np.mean(tr[-k:]))
                        m1 = float(np.mean(tr[-2 * k:-k]))
                        denom = max(1e-8, abs(m1))
                        rel = abs(m2 - m1) / denom
                        loss_stable = rel < rel_thr
                    else:
                        loss_stable = False
                    # ema dice threshold
                    ema_ok = (len(em) > 0) and (float(em[-1]) >= ema_thr)
                    stable = loss_stable or ema_ok
            except Exception:
                stable = False

            if e >= w:
                if not stable and e < hard_start:
                    # extend warmup until stability or hard_start
                    scale = 0.0
                else:
                    # start or continue ramp from configured w
                    scale = _baseline_scale(e)

        self._aux_scale = float(scale)
        try:
            self.print_to_log_file(
                f"[aux_schedule] epoch={e} warmup={w} ramp={r} hard_start={hard_start} dyn={int(dyn_enable)} scale={self._aux_scale:.3f}")
        except Exception:
            pass
