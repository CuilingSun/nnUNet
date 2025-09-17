#!/usr/bin/env python3
import argparse
import os
from typing import List

import numpy as np
import SimpleITK as sitk


def keep_largest_and_min_size(arr: np.ndarray, foreground_labels: List[int], min_size: int, keep_largest: bool) -> np.ndarray:
    # Union of foreground labels into one binary mask
    fg = np.isin(arr, np.array(foreground_labels, dtype=arr.dtype))
    if not np.any(fg):
        return arr
    # Connected components on 3D
    cc = sitk.ConnectedComponent(sitk.GetImageFromArray(fg.astype(np.uint8)))
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)

    # collect component sizes
    comps = list(stats.GetLabels())
    sizes = {lab: int(stats.GetNumberOfPixels(lab)) for lab in comps}

    # sort by size
    comps_sorted = sorted(comps, key=lambda l: sizes[l], reverse=True)

    # build keep mask
    keep = np.zeros_like(fg, dtype=bool)
    kept_any = False
    if keep_largest and len(comps_sorted) > 0:
        largest = comps_sorted[0]
        keep |= sitk.GetArrayFromImage(cc) == largest
        kept_any = True
    for lab in comps_sorted[(1 if keep_largest else 0):]:
        if sizes[lab] >= min_size:
            keep |= sitk.GetArrayFromImage(cc) == lab
            kept_any = True

    if not kept_any:
        # if nothing passes filter and not keeping largest, optionally keep largest anyway to avoid empty seg
        if len(comps_sorted) > 0:
            keep |= sitk.GetArrayFromImage(cc) == comps_sorted[0]

    # apply keep mask: zero out foreground not kept
    out = arr.copy()
    out[fg & (~keep)] = 0
    return out


def main():
    ap = argparse.ArgumentParser(description="Postprocess segmentations: keep LCC and/or remove small components")
    ap.add_argument('--in', dest='inp', required=True, help='Input folder with NIfTI masks (.nii.gz)')
    ap.add_argument('--out', dest='out', required=True, help='Output folder')
    ap.add_argument('--labels', type=int, nargs='*', default=[1], help='Foreground labels to process (default: 1)')
    ap.add_argument('--min-voxels', type=int, default=20, help='Minimum component size to keep (default: 20)')
    ap.add_argument('--keep-largest', action='store_true', help='Keep largest component in addition to min-size filtering')
    ap.add_argument('--file-ending', default='.nii.gz', help='File ending to process (default: .nii.gz)')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    files = [f for f in os.listdir(args.inp) if f.endswith(args.file_ending)]
    for f in files:
        ip = os.path.join(args.inp, f)
        img = sitk.ReadImage(ip)
        arr = sitk.GetArrayFromImage(img)  # z,y,x
        arr = keep_largest_and_min_size(arr, args.labels, args.min_voxels, args.keep_largest)
        out_img = sitk.GetImageFromArray(arr.astype(np.int16))
        out_img.CopyInformation(img)
        op = os.path.join(args.out, f)
        sitk.WriteImage(out_img, op, useCompression=True)


if __name__ == '__main__':
    main()
