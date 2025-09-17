import os
from typing import List, Tuple, Union, Dict

import numpy as np
import torch
from torch import nn

from batchgenerators.utilities.file_and_folder_operations import join, isfile, load_pickle

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMultiEncoderUNetText import (
    nnUNetTrainerMultiEncoderUNetText,
)
from nnunetv2.training.loss.compound_losses import DC_and_topk_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class


class _DataLoaderWithROI(nnUNetDataLoader):
    """
    Extends nnUNetDataLoader to restrict background sampling within an ROI bbox
    (derived from an external mask), with a configurable margin.

    Behavior:
      - If `force_fg` is True, we use standard nnUNet logic to sample around lesion voxels.
      - If `force_fg` is False and an ROI bbox is available, we sample uniformly within the ROI bbox
        (expanded by the configured margin). If unavailable, fall back to default random sampling.
    """

    def __init__(self, *args, roi_dir: Union[str, None] = None,
                 roi_margin_vox: Tuple[int, int, int] = (8, 64, 64), plans_transpose_forward: Tuple[int, int, int] = (0, 1, 2), **kwargs):
        super().__init__(*args, **kwargs)
        self.roi_dir = roi_dir
        self.roi_margin_vox = tuple(int(x) for x in roi_margin_vox)
        self.transpose_forward = plans_transpose_forward
        self._roi_bbox_cache: Dict[str, Union[List[List[int]], None]] = {}

    def _load_roi_bbox_for_case(self, identifier: str, properties: dict, target_shape_after_resampling: Tuple[int, int, int]) -> Union[List[List[int]], None]:
        # cache lookup
        if identifier in self._roi_bbox_cache:
            return self._roi_bbox_cache[identifier]

        if self.roi_dir is None:
            self._roi_bbox_cache[identifier] = None
            return None

        roi_path = join(self.roi_dir, f"{identifier}.nii.gz")
        if not isfile(roi_path):
            self._roi_bbox_cache[identifier] = None
            return None

        # Prefer nibabel for robustness in multiprocessing; fallback to SimpleITK if nib not available
        try:
            import nibabel as nib  # type: ignore
            img = nib.load(roi_path)
            data = np.asanyarray(img.dataobj)
            # ensure channel-less 3D (x,y,z)
            if data.ndim > 3:
                data = np.squeeze(data)
            mask_xyz = (data > 0).astype(np.uint8)
        except Exception:
            try:
                import SimpleITK as sitk  # type: ignore
                itk_img = sitk.ReadImage(roi_path)
                mask_zyx = sitk.GetArrayFromImage(itk_img)  # (z,y,x)
                mask_xyz = np.transpose(mask_zyx, (2, 1, 0)).astype(np.uint8)
            except Exception:
                self._roi_bbox_cache[identifier] = None
                return None

            # apply nnU-Net transpose_forward if not identity (on spatial axes)
            tf = self.transpose_forward
            if tuple(tf) != (0, 1, 2):
                mask_xyz = np.transpose(mask_xyz, tf)

            # crop using the same bbox used during preprocessing
            # bbox format: [[lb_x, ub_x], [lb_y, ub_y], [lb_z, ub_z]]
            bbox = properties.get('bbox_used_for_cropping', None)
            shape_after_crop = properties.get('shape_after_cropping_and_before_resampling', None)
            if bbox is None or shape_after_crop is None:
                self._roi_bbox_cache[identifier] = None
                return None

            slices = tuple(slice(int(lb), int(ub)) for lb, ub in bbox)
            try:
                roi_cropped = mask_xyz[slices]
            except Exception:
                # shape mismatch or invalid bbox -> disable ROI
                self._roi_bbox_cache[identifier] = None
                return None

            # find nonzero bbox in cropped space
            if not np.any(roi_cropped > 0):
                self._roi_bbox_cache[identifier] = None
                return None

            nz = np.nonzero(roi_cropped)
            min_xyz = [int(np.min(ax)) for ax in nz]
            max_xyz = [int(np.max(ax)) for ax in nz]

            # map bbox to the current (resampled) data space via scale factors
            shape_after_crop = tuple(int(x) for x in shape_after_crop)
            tgt = tuple(int(x) for x in target_shape_after_resampling)
            scale = [tgt[d] / max(1, shape_after_crop[d]) for d in range(3)]

            lb = [int(np.floor(min_xyz[d] * scale[d])) for d in range(3)]
            ub = [int(np.ceil((max_xyz[d] + 1) * scale[d])) for d in range(3)]  # +1 because max index is inclusive

            # expand by margin and clip to valid range
            for d in range(3):
                lb[d] = max(0, lb[d] - self.roi_margin_vox[d])
                ub[d] = min(tgt[d], ub[d] + self.roi_margin_vox[d])
                if ub[d] <= lb[d]:
                    ub[d] = min(tgt[d], lb[d] + 1)

            roi_bbox = [[lb[0], ub[0]], [lb[1], ub[1]], [lb[2], ub[2]]]
            self._roi_bbox_cache[identifier] = roi_bbox
            return roi_bbox
        except Exception:
            # any failure -> disable ROI for this case
            self._roi_bbox_cache[identifier] = None
            return None

    def get_bbox(self, data_shape: np.ndarray, force_fg: bool, class_locations: Union[dict, None],
                 overwrite_class: Union[int, Tuple[int, ...]] = None, verbose: bool = False):
        # If foreground is forced, rely on base behavior (sample around lesions)
        if force_fg:
            return super().get_bbox(data_shape, force_fg, class_locations, overwrite_class, verbose)

        # Otherwise, try to restrict background sampling to ROI bbox if available
        # data_shape is a numpy array with (x, y, z)
        need_to_pad = self.need_to_pad.copy()
        dim = len(data_shape)
        for d in range(dim):
            if need_to_pad[d] + data_shape[d] < self.patch_size[d]:
                need_to_pad[d] = self.patch_size[d] - data_shape[d]

        # global bounds in preprocessed space
        lbs = [- need_to_pad[i] // 2 for i in range(dim)]
        ubs = [data_shape[i] + need_to_pad[i] // 2 - self.patch_size[i] for i in range(dim)]

        # fetch ID for current case (last fetched id in generate_train_batch loop)
        # We cannot access identifier directly here, so we compute ROI bbox lazily in generate_train_batch
        # and store it in a temporary attribute. If absent, fallback to default random crop.
        roi_bbox = getattr(self, '_tmp_current_case_roi_bbox', None)
        if roi_bbox is None:
            # fallback to standard behavior for background sampling
            bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]
            bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]
            return bbox_lbs, bbox_ubs

        # compute per-axis sampling bounds constrained to ROI bbox
        roi_lbs = [max(lbs[i], int(roi_bbox[i][0])) for i in range(dim)]
        roi_ubs = [min(ubs[i], int(roi_bbox[i][1]) - self.patch_size[i]) for i in range(dim)]

        # if ROI is smaller than patch, fallback to default random crop
        if any(roi_ubs[i] < roi_lbs[i] for i in range(dim)):
            bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]
            bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]
            return bbox_lbs, bbox_ubs

        bbox_lbs = [np.random.randint(roi_lbs[i], roi_ubs[i] + 1) for i in range(dim)]
        bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]
        return bbox_lbs, bbox_ubs

    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate data & seg (let base class handle shapes and transforms)
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)

        for j, identifier in enumerate(selected_keys):
            force_fg = self.get_do_oversample(j)
            data, seg, seg_prev, properties = self._data.load_case(identifier)
            shape = data.shape[1:]

            # compute ROI bbox for this case once (and cache)
            target_shape_after_resampling = tuple(int(x) for x in shape)
            roi_bbox = self._load_roi_bbox_for_case(identifier, properties, target_shape_after_resampling)
            # stash for get_bbox to use in this iteration
            self._tmp_current_case_roi_bbox = roi_bbox

            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])
            bbox = [[i, j] for i, j in zip(bbox_lbs, bbox_ubs)]

            from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
            data_all[j] = crop_and_pad_nd(data, bbox, 0)

            seg_cropped = crop_and_pad_nd(seg, bbox, -1)
            if seg_prev is not None:
                seg_cropped = np.vstack((seg_cropped, crop_and_pad_nd(seg_prev, bbox, -1)[None]))
            seg_all[j] = seg_cropped

        # cleanup temp
        if hasattr(self, '_tmp_current_case_roi_bbox'):
            delattr(self, '_tmp_current_case_roi_bbox')

        # 2D adapt
        if self.patch_size_was_2d:
            data_all = data_all[:, :, 0]
            seg_all = seg_all[:, :, 0]

        # apply transforms like base class
        if self.transforms is not None:
            import torch as _t
            from threadpoolctl import threadpool_limits
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all = np.ascontiguousarray(data_all, dtype=np.float32)
                    seg_all = np.asarray(seg_all)
                    if seg_all.dtype == np.object_:
                        try:
                            seg_all = np.stack([np.asarray(e) for e in seg_all], axis=0)
                        except Exception:
                            seg_all = np.asarray(seg_all.tolist())
                    if seg_all.ndim == 0:
                        seg_all = seg_all[None]
                    seg_all = np.ascontiguousarray(seg_all)
                    if not np.issubdtype(seg_all.dtype, np.integer):
                        seg_all = seg_all.astype(np.int16, copy=False)

                    images, segs = [], []
                    for b in range(self.batch_size):
                        img_t = _t.as_tensor(np.ascontiguousarray(data_all[b], dtype=np.float32), dtype=_t.float32)
                        seg_np = seg_all[b]
                        if seg_np.dtype == np.object_:
                            try:
                                seg_np = np.stack([np.asarray(e) for e in seg_np], axis=0)
                            except Exception:
                                seg_np = np.asarray(seg_np.tolist())
                        if seg_np.ndim == 0:
                            seg_np = seg_np[None]
                        seg_np = np.ascontiguousarray(seg_np)
                        if not np.issubdtype(seg_np.dtype, np.integer):
                            seg_np = seg_np.astype(np.int16, copy=False)
                        seg_t = _t.as_tensor(seg_np, dtype=_t.int16)

                        tmp = self.transforms(**{'image': img_t, 'segmentation': seg_t})
                        out_img = tmp['image']
                        out_seg = tmp['segmentation']
                        images.append(out_img.contiguous().to(_t.float32) if _t.is_tensor(out_img) else _t.from_numpy(np.ascontiguousarray(out_img, dtype=np.float32)))
                        if isinstance(out_seg, list):
                            segs.append([(s.contiguous() if _t.is_tensor(s) else _t.from_numpy(np.asarray(s))) for s in out_seg])
                        else:
                            segs.append(out_seg.contiguous() if _t.is_tensor(out_seg) else _t.from_numpy(np.asarray(out_seg)))
                    data_all = _t.stack(images)
                    if isinstance(segs[0], list):
                        seg_all = [_t.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                    else:
                        seg_all = _t.stack(segs)
            return {'data': data_all, 'target': seg_all, 'keys': selected_keys}

        return {'data': data_all, 'target': seg_all, 'keys': selected_keys}


class nnUNetTrainerHighRecallROI(nnUNetTrainerMultiEncoderUNetText):
    """
    High-recall trainer with:
      - Strong foreground oversampling (default 0.9)
      - Case-level sampling skew toward positive cases
      - Class-weighted TopK CE + Dice to bias recall
      - Optional ROI-guided background sampling with margin from an external mask folder

    Env knobs:
      NNUNET_OVERSAMPLE_FG: float (default 0.9)
      NNUNET_POS_CASE_WEIGHT: float (default 3.0)  # positive cases are sampled ~W times more often
      NNUNET_CE_W_BG, NNUNET_CE_W_FG: floats (default 0.2, 0.8)
      NNUNET_ROI_DIR: path to NIfTI masks named <identifier>.nii.gz
      NNUNET_ROI_MARGIN: comma ints in voxels, e.g. "8,64,64" (default)
    """

    def initialize(self):
        if self.was_initialized:
            raise RuntimeError('initialize called twice')
        super().initialize()

        # Stronger foreground oversampling by default
        try:
            self.oversample_foreground_percent = float(os.environ.get('NNUNET_OVERSAMPLE_FG', '0.9'))
        except Exception:
            self.oversample_foreground_percent = 0.9

        # Restore sane iteration counts if someone overrode via env in other trainers
        # Favor spending time in training vs validation
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50

        # Mild weight decay is fine
        # self.weight_decay stays as base

    def _build_loss(self):
        # Class-weighted TopK CE + Dice; background down-weighted to favor recall
        ce_w_bg = float(os.environ.get('NNUNET_CE_W_BG', '0.2'))
        ce_w_fg = float(os.environ.get('NNUNET_CE_W_FG', '0.8'))

        n_classes = self.label_manager.num_segmentation_heads
        if n_classes < 2:
            # default to standard loss
            return super()._build_loss()

        if n_classes == 2:
            ce_weight = torch.tensor([ce_w_bg, ce_w_fg], dtype=torch.float32)
        else:
            # distribute foreground weight equally
            fg_each = ce_w_fg / (n_classes - 1)
            ce_weight = torch.tensor([ce_w_bg] + [fg_each] * (n_classes - 1), dtype=torch.float32)

        loss = DC_and_topk_loss(
            {"batch_dice": True, "smooth": 1e-5, "do_bg": False, "ddp": self.is_ddp},
            {"k": 10, "label_smoothing": 0.05, "weight": ce_weight},
            weight_ce=1,
            weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

    def _compute_case_sampling_probabilities(self, dataset_folder: str, identifiers: List[str]) -> Union[List[float], None]:
        """
        Skew case sampling toward positive cases using properties['class_locations'].
        Returns per-identifier probabilities or None if something goes wrong.
        """
        try:
            w_pos = float(os.environ.get('NNUNET_POS_CASE_WEIGHT', '3.0'))
        except Exception:
            w_pos = 3.0

        weights = []
        for k in identifiers:
            pkl = join(dataset_folder, f"{k}.pkl")
            try:
                props = load_pickle(pkl)
                # positive if any class has non-empty voxel list in class_locations
                cl = props.get('class_locations', {})
                is_pos = any(len(v) > 0 for v in cl.values()) if isinstance(cl, dict) else False
                weights.append(w_pos if is_pos else 1.0)
            except Exception:
                # fallback neutral
                weights.append(1.0)

        s = float(sum(weights))
        if s <= 0:
            return None
        return [w / s for w in weights]

    def get_dataloaders(self):
        # copied from base but swap in _DataLoaderWithROI + sampling probabilities
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()
        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        val_transforms = self.get_validation_transforms(
            deep_supervision_scales, is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        # Compute per-case sampling probabilities to upweight positives
        sp_tr = self._compute_case_sampling_probabilities(self.preprocessed_dataset_folder, list(dataset_tr.identifiers))
        sp_val = None  # keep validation unbiased

        # ROI controls via env
        roi_dir = os.environ.get('NNUNET_ROI_DIR', '').strip() or None
        # Fallback to provided gland mask path if env not set
        if roi_dir is None:
            default_gland_dir = '/data2/yyp4247/data/picai_labels/anatomical_delineations/whole_gland/AI/Bosma22b'
            if os.path.isdir(default_gland_dir):
                roi_dir = default_gland_dir
        if roi_dir is not None and not os.path.isdir(roi_dir):
            self.print_to_log_file(f"WARNING: NNUNET_ROI_DIR={roi_dir} not found; disabling ROI cropping")
            roi_dir = None
        roi_margin_env = os.environ.get('NNUNET_ROI_MARGIN', '8,64,64').strip()
        try:
            roi_margin_vox = tuple(int(x) for x in roi_margin_env.replace(' ', '').split(','))
            if len(roi_margin_vox) != 3:
                raise ValueError
        except Exception:
            roi_margin_vox = (2, 32, 32)

        dl_tr = _DataLoaderWithROI(
            dataset_tr, self.batch_size,
            initial_patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=sp_tr, pad_sides=None, transforms=tr_transforms,
            probabilistic_oversampling=True,
            roi_dir=roi_dir,
            roi_margin_vox=roi_margin_vox,
            plans_transpose_forward=tuple(self.plans_manager.transpose_forward),
        )
        dl_val = _DataLoaderWithROI(
            dataset_val, self.batch_size,
            self.configuration_manager.patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=sp_val, pad_sides=None, transforms=val_transforms,
            probabilistic_oversampling=False,
            roi_dir=roi_dir,
            roi_margin_vox=roi_margin_vox,
            plans_transpose_forward=tuple(self.plans_manager.transpose_forward),
        )

        from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
        from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
        from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=None,
                                                        num_processes=allowed_num_processes,
                                                        num_cached=max(6, allowed_num_processes // 2), seeds=None,
                                                        pin_memory=self.device.type == 'cuda', wait_time=0.002)
            mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=dl_val,
                                                      transform=None, num_processes=max(1, allowed_num_processes // 2),
                                                      num_cached=max(3, allowed_num_processes // 4), seeds=None,
                                                      pin_memory=self.device.type == 'cuda',
                                                      wait_time=0.002)

        # pre-warm
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val


class nnUNetTrainerHighRecallROI_Tversky(nnUNetTrainerHighRecallROI):
    """
    Variant that replaces Dice with Tversky (still combined with TopK CE by default).
    Env knobs:
      NNUNET_TVERSKY_ALPHA (default 0.3)
      NNUNET_TVERSKY_BETA  (default 0.7)
      NNUNET_USE_CE (default 1): if 1, we add TopK CE with class weights like base
    """

    def _build_loss(self):
        from nnunetv2.training.loss.tversky import SoftTverskyLoss
        from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
        import numpy as np
        try:
            alpha = float(os.environ.get('NNUNET_TVERSKY_ALPHA', '0.3'))
            beta = float(os.environ.get('NNUNET_TVERSKY_BETA', '0.7'))
        except Exception:
            alpha, beta = 0.3, 0.7

        # Optionally add TopK CE with weights as in base class
        use_ce = str(os.environ.get('NNUNET_USE_CE', '1')).lower() in ('1', 'true', 'yes', 'y')

        # Build Tversky
        tversky = SoftTverskyLoss(batch_dice=True, do_bg=False, smooth=1e-5, ddp=self.is_ddp,
                                  alpha=alpha, beta=beta)

        if use_ce:
            # reuse CE weights from base
            ce_w_bg = float(os.environ.get('NNUNET_CE_W_BG', '0.2'))
            ce_w_fg = float(os.environ.get('NNUNET_CE_W_FG', '0.8'))
            n_classes = self.label_manager.num_segmentation_heads
            if n_classes == 2:
                ce_weight = torch.tensor([ce_w_bg, ce_w_fg], dtype=torch.float32)
            else:
                fg_each = ce_w_fg / (max(1, n_classes - 1))
                ce_weight = torch.tensor([ce_w_bg] + [fg_each] * (n_classes - 1), dtype=torch.float32)
            from nnunetv2.training.loss.robust_ce_loss import TopKLoss
            def combined_loss(net_output, target):
                # Tversky + TopK CE (move class weights to device for CE)
                w = ce_weight.to(net_output.device) if ce_weight.device != net_output.device else ce_weight
                return tversky(net_output, target) + TopKLoss(weight=w, k=10, label_smoothing=0.05)(net_output, target)
            loss = combined_loss
        else:
            loss = tversky

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss


class nnUNetTrainerHighRecallROI_FocalTversky(nnUNetTrainerHighRecallROI):
    """
    Variant using Focal Tversky (recall-focused) optionally plus TopK CE.
    Env knobs:
      NNUNET_TVERSKY_ALPHA (default 0.3)
      NNUNET_TVERSKY_BETA  (default 0.7)
      NNUNET_TVERSKY_GAMMA (default 1.5)
      NNUNET_USE_CE (default 1)
    """

    def _build_loss(self):
        from nnunetv2.training.loss.tversky import FocalTverskyLoss
        from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
        import numpy as np
        try:
            alpha = float(os.environ.get('NNUNET_TVERSKY_ALPHA', '0.3'))
            beta = float(os.environ.get('NNUNET_TVERSKY_BETA', '0.7'))
            gamma = float(os.environ.get('NNUNET_TVERSKY_GAMMA', '1.5'))
        except Exception:
            alpha, beta, gamma = 0.3, 0.7, 1.5

        use_ce = str(os.environ.get('NNUNET_USE_CE', '1')).lower() in ('1', 'true', 'yes', 'y')

        ft = FocalTverskyLoss(batch_dice=True, do_bg=False, smooth=1e-5, ddp=self.is_ddp,
                               alpha=alpha, beta=beta, gamma=gamma)

        if use_ce:
            ce_w_bg = float(os.environ.get('NNUNET_CE_W_BG', '0.2'))
            ce_w_fg = float(os.environ.get('NNUNET_CE_W_FG', '0.8'))
            n_classes = self.label_manager.num_segmentation_heads
            if n_classes == 2:
                ce_weight = torch.tensor([ce_w_bg, ce_w_fg], dtype=torch.float32)
            else:
                fg_each = ce_w_fg / (max(1, n_classes - 1))
                ce_weight = torch.tensor([ce_w_bg] + [fg_each] * (n_classes - 1), dtype=torch.float32)
            from nnunetv2.training.loss.robust_ce_loss import TopKLoss
            def combined_loss(net_output, target):
                w = ce_weight.to(net_output.device) if ce_weight.device != net_output.device else ce_weight
                return ft(net_output, target) + TopKLoss(weight=w, k=10, label_smoothing=0.05)(net_output, target)
            loss = combined_loss
        else:
            loss = ft

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss
