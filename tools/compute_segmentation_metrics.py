#!/usr/bin/env python3
"""Compute segmentation quality metrics for prediction files against ground-truth masks.

This script pairs prediction files with ground-truth files using the prediction file names.
Metrics are computed per-case for the foreground as a whole as well as for individual labels.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

DEFAULT_EXTENSIONS = (".nii.gz", ".nii", ".nrrd", ".mha", ".mhd")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Dice, Precision, Recall, HD95, ASSD, and NSD metrics.")
    parser.add_argument("--pred-dir", required=True, type=Path, help="Directory containing prediction segmentations")
    parser.add_argument("--gt-dir", required=True, type=Path, help="Directory containing ground-truth segmentations")
    parser.add_argument(
        "--labels",
        type=int,
        nargs="*",
        help="Optional list of foreground labels to evaluate. If omitted, labels are inferred from the ground truth.",
    )
    parser.add_argument(
        "--surface-tolerance",
        type=float,
        default=1.0,
        help="Surface tolerance in millimetres for Normalized Surface Dice (default: 1.0).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to save CSV outputs. If omitted, results are only printed to stdout.",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        nargs="*",
        default=None,
        help="File extensions (including dot) to consider. Defaults to common medical image formats.",
    )
    return parser.parse_args()


def canonical_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    return path.stem


def find_prediction_files(directory: Path, extensions: Sequence[str]) -> List[Path]:
    files = []
    for path in directory.iterdir():
        if not path.is_file():
            continue
        if any(path.name.endswith(ext) for ext in extensions):
            files.append(path)
    return sorted(files)


def build_gt_map(directory: Path, extensions: Sequence[str]) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for path in directory.iterdir():
        if not path.is_file():
            continue
        if not any(path.name.endswith(ext) for ext in extensions):
            continue
        stem = canonical_stem(path)
        # keep the first occurrence if duplicates appear; warn in stdout
        if stem in mapping:
            print(f"[WARN] Multiple ground truths share the same stem '{stem}', using '{mapping[stem]}'", file=sys.stderr)
            continue
        mapping[stem] = path
    return mapping


def load_image(path: Path) -> Tuple[np.ndarray, Tuple[float, ...]]:
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)
    spacing = tuple(reversed(image.GetSpacing()))  # align with numpy array axis order (z, y, x)
    return array, spacing


def extract_surface(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    structure = ndimage.generate_binary_structure(mask.ndim, 1)
    eroded = ndimage.binary_erosion(mask, structure=structure, border_value=0)
    surface = mask & ~eroded
    if not np.any(surface):
        surface = mask.copy()
    return surface


def compute_surface_distances(
    reference_mask: np.ndarray, prediction_mask: np.ndarray, spacing: Sequence[float]
) -> Tuple[np.ndarray, np.ndarray]:
    reference_surface = extract_surface(reference_mask)
    prediction_surface = extract_surface(prediction_mask)

    if not np.any(reference_surface) and not np.any(prediction_surface):
        return np.empty(0, dtype=float), np.empty(0, dtype=float)

    distances_ref_to_pred: np.ndarray
    if np.any(prediction_surface):
        dt_pred = ndimage.distance_transform_edt(~prediction_surface, sampling=spacing)
        distances_ref_to_pred = dt_pred[reference_surface]
    else:
        distances_ref_to_pred = np.full(np.count_nonzero(reference_surface), math.inf, dtype=float)

    distances_pred_to_ref: np.ndarray
    if np.any(reference_surface):
        dt_ref = ndimage.distance_transform_edt(~reference_surface, sampling=spacing)
        distances_pred_to_ref = dt_ref[prediction_surface]
    else:
        distances_pred_to_ref = np.full(np.count_nonzero(prediction_surface), math.inf, dtype=float)

    return distances_ref_to_pred, distances_pred_to_ref


def confusion_counts(gt_mask: np.ndarray, pred_mask: np.ndarray) -> Tuple[int, int, int]:
    tp = int(np.count_nonzero(pred_mask & gt_mask))
    fp = int(np.count_nonzero(pred_mask & ~gt_mask))
    fn = int(np.count_nonzero(~pred_mask & gt_mask))
    return tp, fp, fn


def safe_divide(numerator: float, denominator: float, default: float = math.nan) -> float:
    if denominator == 0:
        return default
    return numerator / denominator


def compute_binary_metrics(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    spacing: Sequence[float],
    surface_tolerance: float,
) -> Dict[str, float]:
    gt_present = bool(np.any(gt_mask))
    pred_present = bool(np.any(pred_mask))

    tp, fp, fn = confusion_counts(gt_mask, pred_mask)

    if not gt_present and not pred_present:
        dice = 1.0
        precision = 1.0
        recall = 1.0
        hd95 = 0.0
        assd = 0.0
        nsd = 1.0
    else:
        dice = safe_divide(2 * tp, 2 * tp + fp + fn, default=0.0)
        precision = safe_divide(tp, tp + fp, default=0.0 if pred_present else 1.0)
        recall = safe_divide(tp, tp + fn, default=0.0 if gt_present else 1.0)

        if not gt_present or not pred_present:
            hd95 = math.inf
            assd = math.inf
            nsd = 0.0
        else:
            distances_gt_to_pred, distances_pred_to_gt = compute_surface_distances(gt_mask, pred_mask, spacing)
            all_distances = np.concatenate([distances_gt_to_pred, distances_pred_to_gt])
            if all_distances.size == 0:
                hd95 = 0.0
                assd = 0.0
            else:
                hd95 = float(np.percentile(all_distances, 95))
                assd = float(all_distances.mean())

            if surface_tolerance is None or surface_tolerance < 0:
                nsd = math.nan
            else:
                total_surface_points = distances_gt_to_pred.size + distances_pred_to_gt.size
                if total_surface_points == 0:
                    nsd = 1.0
                else:
                    within_tolerance = (
                        np.count_nonzero(distances_gt_to_pred <= surface_tolerance)
                        + np.count_nonzero(distances_pred_to_gt <= surface_tolerance)
                    )
                    nsd = within_tolerance / total_surface_points

    return {
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "hd95": hd95,
        "assd": assd,
        "nsd": nsd,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def evaluate_case(
    case_id: str,
    gt_array: np.ndarray,
    pred_array: np.ndarray,
    spacing: Sequence[float],
    labels: Iterable[int],
    tolerance: float,
) -> Dict[int, Dict[str, float]]:
    label_metrics: Dict[int, Dict[str, float]] = {}

    # Foreground metrics aggregating all non-zero labels
    foreground_gt = gt_array > 0
    foreground_pred = pred_array > 0
    label_metrics[-1] = compute_binary_metrics(foreground_gt, foreground_pred, spacing, tolerance)

    for label in labels:
        gt_mask = gt_array == label
        pred_mask = pred_array == label
        label_metrics[label] = compute_binary_metrics(gt_mask, pred_mask, spacing, tolerance)

    return label_metrics


def write_case_csv(
    output_path: Path,
    results: List[Tuple[str, int, Dict[str, float]]],
    *,
    include_header: bool = True,
) -> None:
    fieldnames = ["case", "label", "dice", "precision", "recall", "hd95", "assd", "nsd", "tp", "fp", "fn"]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if include_header:
            writer.writeheader()
        for case_id, label, metrics in results:
            row = {
                "case": case_id,
                "label": label,
                "dice": metrics["dice"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "hd95": metrics["hd95"],
                "assd": metrics["assd"],
                "nsd": metrics["nsd"],
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
            }
            writer.writerow(row)


def write_summary_csv(output_path: Path, summary: Dict[int, Dict[str, float]]) -> None:
    fieldnames = ["label", "dice", "precision", "recall", "hd95", "assd", "nsd"]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for label, metrics in summary.items():
            writer.writerow({
                "label": label,
                "dice": metrics["dice"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "hd95": metrics["hd95"],
                "assd": metrics["assd"],
                "nsd": metrics["nsd"],
            })


def format_float(value: float) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return f"{value}"
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def main() -> None:
    args = parse_args()

    extensions = tuple(args.extensions) if args.extensions else DEFAULT_EXTENSIONS

    if not args.pred_dir.is_dir():
        raise FileNotFoundError(f"Prediction directory not found: {args.pred_dir}")
    if not args.gt_dir.is_dir():
        raise FileNotFoundError(f"Ground-truth directory not found: {args.gt_dir}")

    pred_files = find_prediction_files(args.pred_dir, extensions)
    if not pred_files:
        raise RuntimeError(f"No prediction files found in {args.pred_dir} with extensions {extensions}")

    gt_map = build_gt_map(args.gt_dir, extensions)
    if not gt_map:
        raise RuntimeError(f"No ground-truth files found in {args.gt_dir} with extensions {extensions}")

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    labels_to_use: Optional[Sequence[int]] = tuple(sorted(args.labels)) if args.labels else None

    case_results: List[Tuple[str, int, Dict[str, float]]] = []
    summary_accumulators: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    missing_cases = []
    spacing_reference: Optional[Tuple[float, ...]] = None

    for pred_path in pred_files:
        case_id = canonical_stem(pred_path)
        gt_path = gt_map.get(case_id)
        if gt_path is None:
            missing_cases.append(case_id)
            continue

        gt_array, gt_spacing = load_image(gt_path)
        pred_array, pred_spacing = load_image(pred_path)

        if gt_array.shape != pred_array.shape:
            raise ValueError(
                f"Shape mismatch for case '{case_id}': gt shape {gt_array.shape}, pred shape {pred_array.shape}"
            )

        if spacing_reference is None:
            spacing_reference = gt_spacing
        else:
            if not np.allclose(spacing_reference, gt_spacing, atol=1e-5):
                print(
                    f"[WARN] Spacing difference detected for case '{case_id}': {gt_spacing} vs {spacing_reference}",
                    file=sys.stderr,
                )

        if not np.allclose(gt_spacing, pred_spacing, atol=1e-5):
            print(
                f"[WARN] Prediction spacing differs from ground truth for case '{case_id}': {pred_spacing} vs {gt_spacing}",
                file=sys.stderr,
            )

        if labels_to_use is None:
            present_labels = np.unique(gt_array)
            labels = [int(label) for label in present_labels if label != 0]
        else:
            labels = labels_to_use

        metrics_by_label = evaluate_case(case_id, gt_array, pred_array, gt_spacing, labels, args.surface_tolerance)

        for label, metrics in metrics_by_label.items():
            case_results.append((case_id, label, metrics))
            for metric_name in ("dice", "precision", "recall", "hd95", "assd", "nsd"):
                value = metrics[metric_name]
                if isinstance(value, float) and math.isnan(value):
                    continue
                summary_accumulators[label][metric_name].append(value)

    if missing_cases:
        print(
            f"[WARN] The following predictions were skipped because no matching ground truth was found: {', '.join(sorted(missing_cases))}",
            file=sys.stderr,
        )

    if not case_results:
        raise RuntimeError("No cases were evaluated. Ensure file names align between predictions and ground truths.")

    # Compute summary statistics
    summary: Dict[int, Dict[str, float]] = {}
    for label, metrics_lists in summary_accumulators.items():
        summary[label] = {}
        for metric_name, values in metrics_lists.items():
            if not values:
                summary[label][metric_name] = math.nan
                continue
            summary[label][metric_name] = float(np.mean(values))

    # Print to stdout
    print("Case-Level Metrics")
    print("case\tlabel\tdice\tprecision\trecall\thd95\tassd\tnsd")
    for case_id, label, metrics in case_results:
        print(
            f"{case_id}\t{label}\t{format_float(metrics['dice'])}\t{format_float(metrics['precision'])}"
            f"\t{format_float(metrics['recall'])}\t{format_float(metrics['hd95'])}\t{format_float(metrics['assd'])}"
            f"\t{format_float(metrics['nsd'])}"
        )

    print("\nAverage Metrics")
    print("label\tdice\tprecision\trecall\thd95\tassd\tnsd")
    for label, metrics in summary.items():
        print(
            f"{label}\t{format_float(metrics.get('dice', math.nan))}\t{format_float(metrics.get('precision', math.nan))}"
            f"\t{format_float(metrics.get('recall', math.nan))}\t{format_float(metrics.get('hd95', math.nan))}"
            f"\t{format_float(metrics.get('assd', math.nan))}\t{format_float(metrics.get('nsd', math.nan))}"
        )

    if args.output_dir:
        case_csv = args.output_dir / "metrics_per_case.csv"
        summary_csv = args.output_dir / "metrics_summary.csv"
        write_case_csv(case_csv, case_results)
        write_summary_csv(summary_csv, summary)
        print(f"\nSaved per-case metrics to {case_csv}")
        print(f"Saved summary metrics to {summary_csv}")


if __name__ == "__main__":
    main()
