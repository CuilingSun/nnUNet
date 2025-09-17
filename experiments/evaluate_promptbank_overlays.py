"""
Evaluate Sles/Sorg heatmaps against lesion/organ labels, save overlay PNGs,
and write a CSV summary of metrics per case.

Inputs:
- --heatmaps-dir: folder containing Sles_*.nii.gz and Sorg_*.nii.gz pairs
- --lesion-dir: folder with lesion masks (NIfTI); >0 treated as lesion
- --organ-dir: optional folder with prostate masks (NIfTI); >0 treated as organ
- --image-dir: optional folder with raw images (NIfTI) for background overlays

Outputs:
- CSV: metrics.csv under --out-dir
- Overlays: PNGs under --out-dir/overlays/<case>/

Notes:
- All volumes for a given case must share the same spatial shape.
- Case matching is based on basename stem (without extension). For heatmaps,
  expected names are Sles_<stem>.nii.gz and Sorg_<stem>.nii.gz.
"""

import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np


def _scan_dir_nifti(dir_path: str) -> Dict[str, str]:
    mapping = {}
    if dir_path is None:
        return mapping
    for root, _, files in os.walk(dir_path):
        for fn in files:
            if fn.endswith(".nii") or fn.endswith(".nii.gz"):
                base = fn
                stem = base[:-7] if base.endswith(".nii.gz") else base[:-4]
                mapping[stem] = os.path.join(root, fn)
    return mapping


def _scan_heatmaps(dir_path: str) -> List[Tuple[str, str, str]]:
    pairs = []
    sles = {}
    sorg = {}
    for root, _, files in os.walk(dir_path):
        for fn in files:
            if not (fn.endswith(".nii") or fn.endswith(".nii.gz")):
                continue
            if fn.startswith("Sles_"):
                stem = fn[len("Sles_") :]
                stem = stem[:-7] if stem.endswith(".nii.gz") else stem[:-4]
                sles[stem] = os.path.join(root, fn)
            elif fn.startswith("Sorg_"):
                stem = fn[len("Sorg_") :]
                stem = stem[:-7] if stem.endswith(".nii.gz") else stem[:-4]
                sorg[stem] = os.path.join(root, fn)
    common = sorted(set(sles.keys()) & set(sorg.keys()))
    for k in common:
        pairs.append((k, sles[k], sorg[k]))
    return pairs


def _load_nifti(path: str):
    try:
        import nibabel as nib  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("nibabel is required to read NIfTI files.") from e
    img = nib.load(path)
    data = np.asarray(img.get_fdata())
    return img, data


def _minmax01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    vmin = np.nanmin(x)
    vmax = np.nanmax(x)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax - vmin < eps:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - vmin) / (vmax - vmin + eps)).astype(np.float32)


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    am = a.mean()
    bm = b.mean()
    an = a - am
    bn = b - bm
    denom = np.sqrt((an * an).sum() * (bn * bn).sum()) + 1e-12
    return float((an * bn).sum() / denom)


def _coverage(x: np.ndarray, thr: float) -> float:
    x = x.reshape(-1)
    return float((x > thr).mean())


def _top_mass_ratio(x: np.ndarray, q: float = 0.99) -> float:
    x = x.reshape(-1)
    t = np.quantile(x, q)
    s = x[x >= t].sum()
    return float(s / (x.sum() + 1e-8))


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """
    Compute ROC AUC using rank-based method (Mann-Whitney U), handling ties.
    Returns None if only one class present.
    """
    y_true = y_true.astype(bool).reshape(-1)
    y_score = y_score.reshape(-1).astype(np.float64)
    pos = y_true.sum()
    neg = y_true.size - pos
    if pos == 0 or neg == 0:
        return None
    # Rank scores, average ranks for ties
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, y_score.size + 1)
    # Tie handling: find groups with same score
    uniq, inv, counts = np.unique(y_score, return_inverse=True, return_counts=True)
    # average ranks per group
    group_sum = np.bincount(inv, weights=ranks)
    group_avg = group_sum / counts
    ranks = group_avg[inv]
    # Sum of positive ranks
    R_pos = ranks[y_true].sum()
    auc = (R_pos - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auc)


def _dice_at_threshold(pred: np.ndarray, target: np.ndarray, thr: float) -> float:
    p = pred > thr
    t = target.astype(bool)
    inter = np.logical_and(p, t).sum()
    denom = p.sum() + t.sum()
    return float((2.0 * inter) / (denom + 1e-8))


def _best_dice(pred: np.ndarray, target: np.ndarray, n_steps: int = 101) -> Tuple[float, float]:
    best = 0.0
    best_thr = 0.5
    for thr in np.linspace(0.0, 1.0, n_steps):
        d = _dice_at_threshold(pred, target, float(thr))
        if d > best:
            best = d
            best_thr = float(thr)
    return best, best_thr


def _select_slices(lesion: Optional[np.ndarray], sles: np.ndarray, max_slices: int = 6) -> List[int]:
    Z = sles.shape[2]
    if lesion is not None and lesion.any():
        # pick slices with most lesion voxels
        counts = [(z, int(lesion[:, :, z].sum())) for z in range(Z) if lesion[:, :, z].any()]
        counts.sort(key=lambda x: -x[1])
        return [z for z, _ in counts[:max_slices]]
    # fallback: pick slices with highest Sles mass
    masses = [(z, float(sles[:, :, z].sum())) for z in range(Z)]
    masses.sort(key=lambda x: -x[1])
    return [z for z, _ in masses[:max_slices]]


def _overlay_and_save(img_slice: np.ndarray, heat_slice: np.ndarray, lesion: Optional[np.ndarray], organ: Optional[np.ndarray],
                      out_png: str, title: str, cmap: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError("matplotlib is required for overlay PNG saving.") from e

    plt.figure(figsize=(5, 5), dpi=150)
    plt.axis('off')
    # background
    if img_slice is not None:
        bg = _minmax01(img_slice)
    else:
        bg = np.zeros_like(heat_slice, dtype=np.float32)
    plt.imshow(bg, cmap='gray')
    # heatmap
    plt.imshow(heat_slice, cmap=cmap, alpha=0.5, vmin=0.0, vmax=1.0)
    # contours
    if organ is not None and organ.any():
        try:
            plt.contour(organ.astype(bool), colors='cyan', linewidths=0.7)
        except Exception:
            pass
    if lesion is not None and lesion.any():
        try:
            plt.contour(lesion.astype(bool), colors='yellow', linewidths=0.7)
        except Exception:
            pass
    plt.title(title)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, bbox_inches='tight', pad_inches=0)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate and overlay Sles/Sorg heatmaps")
    parser.add_argument("--heatmaps-dir", required=True, help="Directory with Sles_*.nii.gz and Sorg_*.nii.gz")
    parser.add_argument("--lesion-dir", required=True, help="Directory with lesion masks (NIfTI)")
    parser.add_argument("--organ-dir", default=None, help="Directory with organ (prostate) masks (NIfTI)")
    parser.add_argument("--image-dir", default=None, help="Optional directory with raw images for background overlays")
    parser.add_argument("--out-dir", default="./promptbank_eval", help="Output directory for CSV and overlays")
    parser.add_argument("--thr", type=float, default=0.5, help="Threshold for coverage/contrast and Dice@thr")
    parser.add_argument("--max-slices", type=int, default=6, help="Max number of overlay slices per case")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # scan inputs
    pairs = _scan_heatmaps(args.heatmaps_dir)
    if not pairs:
        raise SystemExit("No Sles_/Sorg_ NIfTI pairs found under --heatmaps-dir")
    lesion_map = _scan_dir_nifti(args.lesion_dir)
    organ_map = _scan_dir_nifti(args.organ_dir) if args.organ_dir else {}
    image_map = _scan_dir_nifti(args.image_dir) if args.image_dir else {}

    # CSV
    import csv
    csv_path = os.path.join(args.out_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "case",
            "corr",
            "cov_les@thr",
            "cov_org@thr",
            "top1p_les",
            "top1p_org",
            "les_in_mean",
            "les_out_mean",
            "auc_les_in_gland",
            "dice_sorg@thr",
            "best_dice_sorg",
            "best_thr_sorg",
        ])

        for stem, sles_path, sorg_path in pairs:
            if stem not in lesion_map:
                print(f"Skip {stem}: no lesion mask found")
                continue
            # load
            _, Sles = _load_nifti(sles_path)
            _, Sorg = _load_nifti(sorg_path)
            _, Les = _load_nifti(lesion_map[stem])
            Img = None
            if stem in image_map:
                _, Img = _load_nifti(image_map[stem])
            Org = None
            if stem in organ_map:
                _, Org = _load_nifti(organ_map[stem])

            # sanity shapes
            if Sles.shape != Sorg.shape or Sles.shape != Les.shape:
                print(f"Shape mismatch for {stem}: Sles{Sles.shape}, Sorg{Sorg.shape}, Les{Les.shape}")
                continue

            # normalize heatmaps to [0,1] just in case
            Sles = _minmax01(Sles)
            Sorg = _minmax01(Sorg)

            # metrics
            corr = _pearson_corr(Sles, Sorg)
            cov_les = _coverage(Sles, args.thr)
            cov_org = _coverage(Sorg, args.thr)
            top1p_les = _top_mass_ratio(Sles)
            top1p_org = _top_mass_ratio(Sorg)

            # region masks
            Org_bin = (Org > 0.5) if Org is not None else np.ones_like(Sles, dtype=bool)
            Les_bin = Les > 0.5

            # lesion contrast inside vs outside organ
            les_in = float(Sles[Org_bin].mean()) if Org_bin.any() else float("nan")
            les_out = float(Sles[~Org_bin].mean()) if (~Org_bin).any() else float("nan")

            # AUC of Sles for lesion within gland
            auc = _roc_auc(Les_bin & Org_bin, Sles[Org_bin]) if Org is not None else _roc_auc(Les_bin, Sles)

            # Dice of Sorg vs organ
            dice_thr = float("nan")
            best_dice = float("nan")
            best_thr = float("nan")
            if Org is not None:
                dice_thr = _dice_at_threshold(Sorg, Org_bin, args.thr)
                best_dice, best_thr = _best_dice(Sorg, Org_bin)

            writer.writerow([
                stem,
                f"{corr:.4f}",
                f"{cov_les:.4f}",
                f"{cov_org:.4f}",
                f"{top1p_les:.4f}",
                f"{top1p_org:.4f}",
                f"{les_in:.4f}",
                f"{les_out:.4f}",
                f"{auc:.4f}" if auc is not None else "nan",
                f"{dice_thr:.4f}" if not np.isnan(dice_thr) else "nan",
                f"{best_dice:.4f}" if not np.isnan(best_dice) else "nan",
                f"{best_thr:.3f}" if not np.isnan(best_thr) else "nan",
            ])

            # overlays
            out_case_dir = os.path.join(args.out_dir, "overlays", stem)
            # choose slices
            slices = _select_slices(Les_bin, Sles, max_slices=args.max_slices)
            # choose background image
            if Img is not None and Img.ndim == 3:
                img_vol = Img
            else:
                img_vol = None

            for z in slices:
                img_slice = img_vol[:, :, z] if img_vol is not None else None
                _overlay_and_save(
                    img_slice,
                    Sles[:, :, z],
                    Les_bin[:, :, z],
                    Org_bin[:, :, z] if Org is not None else None,
                    os.path.join(out_case_dir, f"{stem}_z{z:03d}_Sles.png"),
                    title=f"{stem} z={z} Sles (lesion=yellow, organ=cyan)",
                    cmap="hot",
                )
                _overlay_and_save(
                    img_slice,
                    Sorg[:, :, z],
                    Les_bin[:, :, z],
                    Org_bin[:, :, z] if Org is not None else None,
                    os.path.join(out_case_dir, f"{stem}_z{z:03d}_Sorg.png"),
                    title=f"{stem} z={z} Sorg (lesion=yellow, organ=cyan)",
                    cmap="Greens",
                )

    print(f"Saved CSV: {csv_path}")
    print(f"Saved overlays under: {os.path.join(args.out_dir, 'overlays')}")


if __name__ == "__main__":
    main()

