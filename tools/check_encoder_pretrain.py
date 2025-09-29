#!/usr/bin/env python3
"""Inspect whether a checkpoint provides compatible encoder weights for MultiEncoderUNet trainers."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from batchgenerators.utilities.file_and_folder_operations import isdir, join, load_json

from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.nets.MultiEncoderUNet import (
    get_multi_encoder_unet_2d_from_plans,
    get_multi_encoder_unet_3d_from_plans,
)
from nnunetv2.nets.MultiEncodernnUNet import build_multi_encoder_from_plans


PREFIX_CANDIDATES: Tuple[str, ...] = (
    "module.",
    "network.",
    "model.",
    "state_dict.",
    "nnunet.",
    "seg_network.",
)


def strip_common_prefix(state: Dict[str, torch.Tensor], prefixes: Iterable[str] = PREFIX_CANDIDATES) -> Dict[str, torch.Tensor]:
    """Remove a shared leading prefix (e.g. 'module.') if *all* keys contain it."""
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


def summarize_prefixes(keys: Iterable[str], depth: int = 2, topk: int = 8) -> List[Tuple[str, int]]:
    ctr = Counter()
    for key in keys:
        parts = key.split('.')
        prefix = '.'.join(parts[:depth])
        ctr[prefix] += 1
    return ctr.most_common(topk)


def is_seg_or_head(key: str) -> bool:
    return (
        ".seg_layers." in key
        or key.endswith(".seg_layers")
        or "final_conv" in key
        or "output_block" in key
    )


def extract_state_dict(ckpt_path: Path) -> Dict[str, torch.Tensor]:
    saved = torch.load(str(ckpt_path), map_location='cpu')
    if isinstance(saved, dict):
        for candidate in ("network_weights", "state_dict", "model_state", "model_state_dict"):
            val = saved.get(candidate)
            if isinstance(val, dict):
                saved = val
                break
    if not isinstance(saved, dict):
        raise RuntimeError(f"Checkpoint {ckpt_path} does not contain a recognizable state dict.")
    tensors = {k: v for k, v in saved.items() if isinstance(v, torch.Tensor)}
    if not tensors:
        raise RuntimeError(f"Checkpoint {ckpt_path} does not contain any tensor weights.")
    tensors = strip_common_prefix(tensors)
    return tensors


def assign_tensor(
    state: Dict[str, torch.Tensor],
    target_sd: Dict[str, torch.Tensor],
    source_key: str,
    target_key: str,
    matched: Dict[str, torch.Tensor],
    used_source: set,
    shape_mismatch: List[Tuple[str, str, Tuple[int, ...], Tuple[int, ...]]],
    missing_target: List[Tuple[str, str, str]],
    note: str | None = None,
) -> bool:
    tgt_tensor = target_sd.get(target_key)
    if tgt_tensor is None:
        missing_target.append((source_key, target_key, 'not_in_target'))
        return False
    tensor = state[source_key]
    if tensor.shape != tgt_tensor.shape:
        shape_mismatch.append((source_key, target_key, tuple(tensor.shape), tuple(tgt_tensor.shape)))
        return False
    matched[target_key] = tensor
    used_source.add(source_key)
    return True


def map_encoder_weights(state: Dict[str, torch.Tensor], target_sd: Dict[str, torch.Tensor]):
    matched: Dict[str, torch.Tensor] = {}
    shape_mismatch: List[Tuple[str, str, Tuple[int, ...], Tuple[int, ...]]] = []
    missing_target: List[Tuple[str, str, str]] = []
    fallback_hits: List[Tuple[str, str, str]] = []
    used_source: set = set()

    src_keys = list(state.keys())
    has_multi = any(k.startswith('encoders.0.') for k in src_keys)
    has_plain = any(k.startswith('encoder.') for k in src_keys)

    if has_multi:
        for sk in src_keys:
            if not sk.startswith('encoders.0.') or is_seg_or_head(sk):
                continue
            assign_tensor(state, target_sd, sk, sk, matched, used_source, shape_mismatch, missing_target)
    elif has_plain:
        prefix = 'encoder.'
        for sk in src_keys:
            if not sk.startswith(prefix) or is_seg_or_head(sk):
                continue
            tk = 'encoders.0.' + sk[len(prefix):]
            tgt_tensor = target_sd.get(tk)
            if tgt_tensor is None:
                missing_target.append((sk, tk, 'not_in_target'))
                continue
            tensor = state[sk]
            if (
                tensor.ndim == tgt_tensor.ndim
                and tensor.shape[0] == tgt_tensor.shape[0]
                and tensor.shape[2:] == tgt_tensor.shape[2:]
                and tgt_tensor.shape[1] == 1
                and tensor.shape[1] != 1
            ):
                tensor = tensor.mean(dim=1, keepdim=True)
            if tensor.shape != tgt_tensor.shape:
                shape_mismatch.append((sk, tk, tuple(tensor.shape), tuple(tgt_tensor.shape)))
                continue
            matched[tk] = tensor
            used_source.add(sk)
    else:
        # heuristic fallback: match by identical tensor shape for encoder-0 weights
        for tk, tgt_tensor in target_sd.items():
            if not tk.startswith('encoders.0.') or is_seg_or_head(tk):
                continue
            for sk in src_keys:
                if sk in used_source:
                    continue
                tensor = state[sk]
                if tensor.shape == tgt_tensor.shape:
                    if assign_tensor(state, target_sd, sk, tk, matched, used_source, shape_mismatch, missing_target):
                        fallback_hits.append((sk, tk, 'shape_match'))
                        break

    return matched, shape_mismatch, missing_target, used_source, fallback_hits, has_multi, has_plain




from collections import OrderedDict

def _convert_legacy_to_multi(network, legacy_state):
    target_state = network.state_dict()
    converted = OrderedDict()
    encoder_count = len(getattr(network, 'encoders', []))
    for key, tensor in legacy_state.items():
        if not key.startswith('encoder.stages.'):
            continue
        parts = key.split('.')
        if len(parts) < 7 or parts[4] != 'convs' or parts[6] == 'all_modules':
            continue
        stage_idx, conv_idx = parts[2], parts[5]
        attr = '.'.join(parts[6:])
        new_tail = f'stages.{stage_idx}.convs.{conv_idx}.{attr}'
        target_key = f'encoders.0.{new_tail}'
        target_tensor = target_state.get(target_key)
        if target_tensor is None:
            continue
        target_shape = target_tensor.shape
        mapped = None
        if tensor.shape == target_shape:
            mapped = tensor
        elif (
            tensor.ndim == len(target_shape)
            and tensor.shape[0] == target_shape[0]
            and tensor.shape[1] != target_shape[1]
            and target_shape[1] == 1
            and tensor.shape[2:] == target_shape[2:]
        ):
            mapped = tensor.mean(dim=1, keepdim=True)
        elif (
            tensor.ndim == len(target_shape)
            and tensor.shape[0] == target_shape[0]
            and tensor.shape[1] == target_shape[1]
            and tensor.shape[3:] == target_shape[3:]
            and target_shape[2] == 1
        ):
            mapped = tensor.mean(dim=2, keepdim=True)
        if mapped is None:
            continue
        for enc_idx in range(encoder_count):
            converted[f'encoders.{enc_idx}.{new_tail}'] = mapped.clone()
    return converted
def build_network(dataset: str, configuration: str, plans_identifier: str, variant: str):
    if nnUNet_preprocessed is None:
        suggestion = (
            'export nnUNet_raw="${nnUNet_raw:-/data2/yyp4247/data/nnUNet_raw}"\n'
            'export nnUNet_preprocessed="${nnUNet_preprocessed:-/data2/yyp4247/data/nnUNet_preprocessed}"'
        )
        raise RuntimeError(
            'Environment variable nnUNet_preprocessed is not defined.\n'
            'Set it before running, for example:\n'
            + suggestion
        )
    dataset_name = maybe_convert_to_dataset_name(dataset)
    base = join(nnUNet_preprocessed, dataset_name)
    if not isdir(base):
        raise RuntimeError(f'Preprocessed dataset folder not found: {base}')

    plans_path = join(base, f'{plans_identifier}.json')
    dataset_json_path = join(base, 'dataset.json')

    plans_manager = PlansManager(plans_path)
    configuration_manager = plans_manager.get_configuration(configuration)
    dataset_json = load_json(dataset_json_path)

    num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
    dim = len(configuration_manager.patch_size)

    if variant == 'fusion':
        network = build_multi_encoder_from_plans(
            plans_manager, dataset_json, configuration_manager, num_input_channels, deep_supervision=True
        )
    else:
        if dim == 3:
            network = get_multi_encoder_unet_3d_from_plans(
                plans_manager,
                dataset_json,
                configuration_manager,
                num_input_channels,
                deep_supervision=True,
            )
        elif dim == 2:
            network = get_multi_encoder_unet_2d_from_plans(
                plans_manager,
                dataset_json,
                configuration_manager,
                num_input_channels,
                deep_supervision=True,
            )
        else:
            raise RuntimeError(f'Unsupported patch dimensionality: {dim}')

    return network, plans_manager, configuration_manager, dataset_json


def main():
    parser = argparse.ArgumentParser(
        description='Check compatibility of encoder pretrained weights with MultiEncoderUNet trainers.'
    )
    parser.add_argument('checkpoint', type=Path, help='Path to the .pth checkpoint to inspect.')
    parser.add_argument('--dataset', default='Dataset2202_picai_split', help='Dataset name or numeric id.')
    parser.add_argument('--configuration', default='3d_fullres', help='Configuration to instantiate (default: 3d_fullres).')
    parser.add_argument('--plans', default='nnUNetPlans', help='Plans identifier (default: nnUNetPlans).')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps'], help='Device for instantiation (default: cpu).')
    parser.add_argument('--variant', default='fusion', choices=['fusion', 'legacy'],
                        help='fusion -> new MultiEncodernnUNet (default), legacy -> old MultiEncoderUNet')
    parser.add_argument('--max-list', type=int, default=20, help='Maximum number of detailed entries to print per category.')
    args = parser.parse_args()

    ckpt_path = args.checkpoint.expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available.')

    torch.set_grad_enabled(False)

    network, plans_manager, configuration_manager, _ = build_network(
        args.dataset, args.configuration, args.plans, args.variant
    )
    network.to(torch.device(args.device))
    target_sd = network.state_dict()

    state = extract_state_dict(ckpt_path)
    if args.variant == 'fusion':
        if any(k.startswith('encoder.') for k in state):
            mapped = _convert_legacy_to_multi(network, state)
        else:
            mapped = state
        load_res = network.load_state_dict(mapped, strict=False)
        missing = list(getattr(load_res, 'missing_keys', []))
        unexpected = list(getattr(load_res, 'unexpected_keys', []))
        print('=== load_state_dict summary ===')
        print(f'Missing keys: {len(missing)}')
        if missing:
            for name in missing[:args.max_list]:
                print(f'  missing -> {name}')
            if len(missing) > args.max_list:
                print(f'  ... ({len(missing) - args.max_list} more)')
        print(f'Unexpected keys: {len(unexpected)}')
        if unexpected:
            for name in unexpected[:args.max_list]:
                print(f'  unexpected -> {name}')
            if len(unexpected) > args.max_list:
                print(f'  ... ({len(unexpected) - args.max_list} more)')
        return
    mapped, shape_mismatch, missing_target, used_source, fallback_hits, has_multi, has_plain = map_encoder_weights(
        state, target_sd
    )

    load_res = network.load_state_dict(mapped, strict=False)

    encoder_keys = [k for k in target_sd if k.startswith('encoders.')]
    encoder_indices = sorted({k.split('.')[1] for k in encoder_keys})

    print('=== Target network ===')
    print(f'Encoders present: {encoder_indices} (total tensors: {len(target_sd)})')
    first_conv = next((k for k in encoder_keys if k.endswith('conv1.weight')), None)
    if first_conv:
        print(f'First encoder conv key: {first_conv}, shape: {tuple(target_sd[first_conv].shape)}')

    print('\n=== Checkpoint stats ===')
    print(f'File: {ckpt_path}')
    print(f'Tensor entries in checkpoint: {len(state)}')
    print('Top key prefixes:')
    for prefix, count in summarize_prefixes(state.keys(), depth=2, topk=8):
        print(f'  {prefix}: {count}')
    layout = 'encoders.0.*' if has_multi else ('encoder.*' if has_plain else 'unknown')
    print(f'Detected encoder key layout: {layout}')

    print('\n=== Mapping summary ===')
    print(f'Matched tensors: {len(mapped)}')
    missing_keys = list(getattr(load_res, 'missing_keys', []))
    unexpected_keys = list(getattr(load_res, 'unexpected_keys', []))
    print(f'load_state_dict missing keys: {len(missing_keys)}')
    for name in missing_keys[:args.max_list]:
        print(f'  missing -> {name}')
    if len(missing_keys) > args.max_list:
        print(f'  ... ({len(missing_keys) - args.max_list} more)')
    print(f'Unexpected checkpoint keys: {len(unexpected_keys)}')
    for name in unexpected_keys[:args.max_list]:
        print(f'  unexpected -> {name}')
    if len(unexpected_keys) > args.max_list:
        print(f'  ... ({len(unexpected_keys) - args.max_list} more)')

    if shape_mismatch:
        print(f'\nShape mismatches ({len(shape_mismatch)}):')
        for src, tgt, s_shape, t_shape in shape_mismatch[:args.max_list]:
            print(f'  {src} -> {tgt}: src {s_shape} vs tgt {t_shape}')
        if len(shape_mismatch) > args.max_list:
            print(f'  ... ({len(shape_mismatch) - args.max_list} more)')

    if missing_target:
        print(f'\nMissing target entries ({len(missing_target)}):')
        for src, tgt, reason in missing_target[:args.max_list]:
            print(f'  {src} -> {tgt} ({reason})')
        if len(missing_target) > args.max_list:
            print(f'  ... ({len(missing_target) - args.max_list} more)')

    unused = sorted(set(state.keys()) - used_source)
    if unused:
        print(f'\nCheckpoint tensors not used ({len(unused)}):')
        for key in unused[:args.max_list]:
            print(f'  {key}')
        if len(unused) > args.max_list:
            print(f'  ... ({len(unused) - args.max_list} more)')

    if fallback_hits:
        print(f'\nFallback shape matches applied ({len(fallback_hits)}):')
        for src, tgt, note in fallback_hits[:args.max_list]:
            print(f'  {src} -> {tgt} ({note})')
        if len(fallback_hits) > args.max_list:
            print(f'  ... ({len(fallback_hits) - args.max_list} more)')


if __name__ == '__main__':
    main()
