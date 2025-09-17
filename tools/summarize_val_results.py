#!/usr/bin/env python3
import argparse
import json
import os
from glob import glob
from typing import List, Tuple


def find_summary_jsons(root: str) -> List[Tuple[str, str]]:
    paths = []
    for fold in ('fold_0', 'fold_all', 'fold_1', 'fold_2', 'fold_3', 'fold_4'):
        # base validation
        for p in glob(os.path.join(root, '**', fold, 'validation', 'summary.json'), recursive=True):
            paths.append((os.path.dirname(p), p))
        # postprocessed variants
        for p in glob(os.path.join(root, '**', fold, 'validation_pp_*', 'summary.json'), recursive=True):
            paths.append((os.path.dirname(p), p))
    # de-duplicate
    uniq = {}
    for d, p in paths:
        uniq[p] = d
    return [(v, k) for k, v in uniq.items()]


def infer_label_from_path(path: str) -> str:
    base = os.path.basename(path)
    if base.startswith('validation_pp_'):
        return base.replace('validation_pp_', '')
    return 'validation'


def main():
    ap = argparse.ArgumentParser(description='Summarize nnUNet validation metrics across models/folders')
    ap.add_argument('--root', required=True, help='Root under nnUNet_results/DatasetXXX_* to scan')
    ap.add_argument('--out', default='summary.csv', help='Output CSV file')
    args = ap.parse_args()

    items = find_summary_jsons(args.root)
    rows = []
    for d, p in items:
        try:
            with open(p, 'r') as f:
                data = json.load(f)
            fg_mean = data.get('foreground_mean', {})
            dice = fg_mean.get('Dice', None)
            iou = fg_mean.get('IoU', None)
            label = infer_label_from_path(d)
            rows.append((d, label, dice, iou))
        except Exception:
            continue

    # write CSV
    with open(args.out, 'w') as f:
        f.write('folder,label,foreground_mean_dice,foreground_mean_iou\n')
        for d, label, dice, iou in rows:
            f.write(f'{d},{label},{dice if dice is not None else ""},{iou if iou is not None else ""}\n')
    print(f'Wrote {len(rows)} rows to {args.out}')


if __name__ == '__main__':
    main()

