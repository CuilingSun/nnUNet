import os
from typing import List, Union

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

    def _compute_aux_losses(self, extras: dict, target: Union[List[torch.Tensor], torch.Tensor]):
        if extras is None or (self.lambda_align == 0 and self.lambda_heat == 0):
            return 0.0

        # get highest-resolution seg target if DS
        if isinstance(target, list):
            gt = target[0]
        else:
            gt = target

        # assume 'sim' is in [0,1] after sigmoid and possibly upsampled to match seg size
        sim = extras.get('sim', None)
        if sim is None:
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
            return self.lambda_align * l_align
        with torch.autocast(self.device.type, enabled=False) if self.device.type == 'cuda' else dummy_context():
            bce = nn.functional.binary_cross_entropy(sim, fg_mask)

        # apply curriculum scaling if configured
        scale = getattr(self, '_aux_scale', 1.0)
        la = self.lambda_align * float(scale)
        lh = self.lambda_heat * float(scale)
        return la * l_align + lh * bce

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

            l = self.loss(seg, target)
            if text is not None and extras is not None:
                l = l + self._compute_aux_losses(extras, target)

        # skip non-finite loss to avoid poisoning optimizer state
        if not torch.isfinite(l):
            self.print_to_log_file('WARNING: non-finite loss detected in text trainer; skipping optimizer step')
            self.optimizer.zero_grad(set_to_none=True)
            return {'loss': float(0.0)}

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}

    def on_epoch_start(self):
        # keep base logging and timers
        super().on_epoch_start()
        # update auxiliary loss scale according to warmup/ramp schedule
        warm = getattr(self, 'aux_warmup_epochs', 0)
        ramp = getattr(self, 'aux_ramp_epochs', 0)
        if warm == 0 and ramp == 0:
            self._aux_scale = 1.0
            return
        e = int(self.current_epoch)
        if e < warm:
            new_scale = 0.0
        elif e < (warm + ramp) and ramp > 0:
            # linear ramp from 0 -> 1 across ramp epochs
            new_scale = float(e - warm + 1) / float(max(1, ramp))
            new_scale = max(0.0, min(1.0, new_scale))
        else:
            new_scale = 1.0
        if abs(new_scale - float(getattr(self, '_aux_scale', 0.0))) > 1e-6:
            self._aux_scale = new_scale
            self.print_to_log_file(
                f"Aux loss scale updated to {self._aux_scale:.3f} (warmup={warm}, ramp={ramp})")

    def validation_step(self, batch: dict) -> dict:
        # This mirrors nnUNetTrainer.validation_step but passes text to the network
        from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Prepare text for this batch
        text = self.text_embed
        if text is not None:
            if text.shape[0] == 1 and data.shape[0] > 1:
                text = text.expand(data.shape[0], -1)
            else:
                text = text[: data.shape[0]]
        elif hasattr(self, 'text_embed_param'):
            text = self.text_embed_param[:1].expand(data.shape[0], -1)

        with torch.autocast(self.device.type, enabled=self.use_amp) if self.device.type == 'cuda' else dummy_context():
            out = None
            try:
                output = self.network(data, text=text)
            except TypeError:
                output = self.network(data)
            # do not use extras in loss here; keep validation loss comparable
            l = self.loss(output, target)

        # online evaluation (pseudo dice)
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        axes = [0] + list(range(2, output.ndim))
        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)
        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        # Optionally save heatmaps (no_grad) for inspection
        if self.save_heatmaps:
            with torch.no_grad():
                keys = batch.get('keys', None)
                try:
                    out = self.network(data, text=text, return_extra=True)
                except TypeError:
                    out = None
                if isinstance(out, dict) and 'sim' in out:
                    sim = out['sim'].detach().cpu()
                    img = data.detach().cpu()
                    out_dir = join(self.output_folder, 'val_heatmaps', f'epoch_{self.current_epoch:03d}')
                    maybe_mkdir_p(out_dir)
                    b = sim.shape[0]
                    for bi in range(b):
                        if sim.ndim == 5:
                            _, _, z, y, x = sim.shape
                            mid = z // 2
                            heat = sim[bi, 0, mid].numpy()
                            bg = img[bi, 0, mid].numpy()
                        else:
                            heat = sim[bi, 0].numpy()
                            bg = img[bi, 0].numpy()
                        name = keys[bi] if keys is not None and bi < len(keys) else f'sample{bi}'
                        path = join(out_dir, f'{name}.png')
                        plt.figure(figsize=(5, 5))
                        plt.imshow(bg, cmap='gray')
                        plt.imshow(heat, cmap='hot', alpha=0.5, vmin=0, vmax=1)
                        plt.axis('off')
                        plt.tight_layout()
                        plt.savefig(path, dpi=150)
                        plt.close()

        return {'loss': float(l.detach().cpu().item()), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}

    # ---------- Full-volume heatmap export ----------
    @torch.inference_mode()
    def _predict_full_heatmap(self, data: torch.Tensor) -> torch.Tensor:
        """Aggregate text heatmap over the full case using sliding-window prediction.
        Returns a tensor of shape [1, *image_size] on CPU.
        """
        self.network.eval()
        device = self.device
        # pad to patch size
        data, slicer_revert_padding = pad_nd_image(data, self.configuration_manager.patch_size,
                                                   'constant', {'value': 0}, True,
                                                   None)

        image_size = data.shape[1:]
        tile_size = tuple(self.configuration_manager.patch_size)
        steps = compute_steps_for_sliding_window(image_size, tile_size, 0.5)
        # build slicers
        slicers = []
        for z in steps[0 if len(image_size)==1 else 0]:
            pass  # dummy to appease flake
        # general slicer generation
        import itertools
        for coords in itertools.product(*steps):
            sl = (slice(None),) + tuple(slice(int(c), int(c)+int(t)) for c, t in zip(coords, tile_size))
            slicers.append(sl)

        # preallocate
        pred = torch.zeros((1, *image_size), dtype=torch.float16, device=device)
        nmap = torch.zeros(image_size, dtype=torch.float16, device=device)
        if self.inference_allowed_mirroring_axes is not None:
            mirror_axes = [m + 2 for m in self.inference_allowed_mirroring_axes]
        else:
            mirror_axes = None
        gauss = compute_gaussian(tile_size, sigma_scale=1./8, value_scaling_factor=10, device=device)

        for sl in slicers:
            x = torch.clone(data[sl][None], memory_format=torch.contiguous_format).to(device)
            # forward
            try:
                out = self.network(x, text=(self.text_embed[:1] if self.text_embed is not None else None), return_extra=True)
            except TypeError:
                out = None
            if isinstance(out, dict) and 'sim' in out:
                h = out['sim'][0].to(device)  # [1, *tile]
            else:
                # if heatmap unavailable, fill zeros
                h = torch.zeros((1, *tile_size), dtype=torch.float16, device=device)

            # TTA mirroring if desired
            if mirror_axes is not None:
                import itertools as it
                for axes in [c for i in range(len(mirror_axes)) for c in it.combinations(mirror_axes, i + 1)]:
                    try:
                        tmp = self.network(torch.flip(x, axes), text=(self.text_embed[:1] if self.text_embed is not None else None), return_extra=True)
                        if isinstance(tmp, dict) and 'sim' in tmp:
                            h += torch.flip(tmp['sim'][0].to(device), axes)
                    except TypeError:
                        pass
                h /= (1 + len([c for i in range(len(mirror_axes)) for c in it.combinations(mirror_axes, i + 1)]))

            h = h * gauss  # weighting
            pred[sl] += h
            nmap[sl[1:]] += gauss

        torch.div(pred, nmap, out=pred)
        pred = pred[(slice(None), *slicer_revert_padding[1:])].to('cpu')
        return pred

    def _save_full_heatmap_npz(self, heatmap: torch.Tensor, properties: dict, ofile_truncated: str):
        """Saves two npz files: preprocessed space and original image space.
        If NNUNET_EXPORT_HEATMAP_NIFTI is enabled, also writes an orig-space NIfTI.
        """
        import numpy as np
        import os
        hm = heatmap.numpy().astype(np.float32)
        np.savez_compressed(ofile_truncated + '_pproc_heatmap.npz', heatmap=hm)

        # try to map back to original image space (like probabilities path)
        try:
            # reuse the export pipeline for resampling & cropping reversion
            from nnunetv2.utilities.label_handling.label_handling import LabelManager
            label_manager = self.plans_manager.get_label_manager(self.dataset_json)
            spacing_transposed = [properties['spacing'][i] for i in self.plans_manager.transpose_forward]
            current_spacing = self.configuration_manager.spacing if \
                len(self.configuration_manager.spacing) == len(properties['shape_after_cropping_and_before_resampling']) else \
                [spacing_transposed[0], *self.configuration_manager.spacing]

            res = self.configuration_manager.resampling_fn_probabilities(
                heatmap, properties['shape_after_cropping_and_before_resampling'], current_spacing,
                [properties['spacing'][i] for i in self.plans_manager.transpose_forward]
            )
            if isinstance(res, torch.Tensor):
                res = res.cpu().numpy()
            # revert cropping & transpose like probabilities path
            probs_rev = label_manager.revert_cropping_on_probabilities(
                torch.from_numpy(res), properties['bbox_used_for_cropping'], properties['shape_before_cropping'])
            if isinstance(probs_rev, torch.Tensor):
                probs_rev = probs_rev.numpy()
            probs_rev = probs_rev.transpose([0] + [i + 1 for i in self.plans_manager.transpose_backward])
            probs_rev = probs_rev.astype(np.float32)
            np.savez_compressed(ofile_truncated + '_orig_heatmap.npz', heatmap=probs_rev)

            # Optional NIfTI export (orig space)
            if self.export_heatmap_nifti:
                try:
                    import SimpleITK as sitk
                    sitk_info = properties.get('sitk_stuff', {})
                    # probs_rev shape: (C, X, Y, Z). Use channel 0
                    arr_xyz = probs_rev[0]
                    # sitk expects (z,y,x)
                    arr_zyx = np.transpose(arr_xyz, (2, 1, 0))
                    img = sitk.GetImageFromArray(arr_zyx)
                    if 'spacing' in sitk_info:
                        img.SetSpacing(tuple(float(s) for s in sitk_info['spacing']))
                    if 'origin' in sitk_info:
                        img.SetOrigin(tuple(float(o) for o in sitk_info['origin']))
                    if 'direction' in sitk_info:
                        dirv = sitk_info['direction']
                        img.SetDirection(tuple(float(d) for d in dirv))
                    sitk.WriteImage(img, ofile_truncated + '_orig_heatmap.nii.gz', useCompression=True)
                except Exception as e:
                    self.print_to_log_file(f'WARNING: Failed to save NIfTI heatmap for {ofile_truncated}: {e}')
        except Exception as e:
            self.print_to_log_file(f'WARNING: Failed to save orig-space heatmap for {ofile_truncated}: {e}')

    def perform_actual_validation(self, save_probabilities: bool = False):
        # first run the default validation (saves segmentations and metrics)
        super().perform_actual_validation(save_probabilities)

        # optional: export full-case heatmaps
        if not self.export_full_heatmap or self.text_embed is None:
            return

        self.network.eval()
        # collect val keys similar to base implementation
        _, val_keys = self.do_split()
        dataset_val = self.dataset_class(self.preprocessed_dataset_folder, val_keys)

        out_base = join(self.output_folder, 'validation', self.heatmap_dirname)
        maybe_mkdir_p(out_base)
        for k in dataset_val.identifiers:
            self.print_to_log_file(f"exporting full heatmap {k}")
            data, _, _, properties = dataset_val.load_case(k)
            data = torch.from_numpy(data[:])  # [C, ...]
            hm = self._predict_full_heatmap(data)  # [1, ...] cpu
            self._save_full_heatmap_npz(hm, properties, join(out_base, k))
