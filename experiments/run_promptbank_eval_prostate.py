"""
Minimal CLI to compute and save Sles/Sorg heatmaps from NIfTI feature volumes
using a fixed Prompt Bank (lesion & organ) for prostate.

Inputs:
- One or more NIfTI feature volumes (.nii or .nii.gz), each 4D with channels D.
  By default we assume channels-last layout: [X, Y, Z, D]. If your features are
  channels-first [D, X, Y, Z], pass --channels-first.
- Text prompt vectors either:
  (A) Pre-encoded as two .npy files: --t-les, --t-org (shapes [K_les,D], [K_org,D])
  (B) Encoded on the fly with OpenCLIP: --openclip MODEL PRETRAINED (optional).
      Use only if your features’ channel dimension D matches the selected model’s
      text embedding dim and the A1 projection was trained for that.

Outputs:
- Sles_*.nii.gz and Sorg_*.nii.gz saved next to --out-dir (default ./promptbank_out).
- Optionally also save .npy via --save-npy.

Processing model:
- Processes inputs sequentially (per-file) to support heterogeneous spatial
  shapes and reduce peak memory. Performs an early D-dimension check on the
  first file to fail fast if text/feature dims mismatch.

This script does not modify any existing project files. It relies on
experiments/promptbank_eval_prostate_test.py.
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple, Optional

# Make script runnable via `python experiments/run_promptbank_eval_prostate.py` by
# enabling fallback relative import when top-level package is not on sys.path.
try:
    from experiments.promptbank_eval_prostate_test import (
        build_prostate_prompt_bank,
        PromptBankEvaluator,
        save_heatmaps_nifti,
        save_heatmaps_npy,
    )
except ModuleNotFoundError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from promptbank_eval_prostate_test import (  # type: ignore
        build_prostate_prompt_bank,
        PromptBankEvaluator,
        save_heatmaps_nifti,
        save_heatmaps_npy,
    )

import numpy as np


def _load_nifti(path: str):
    try:
        import nibabel as nib  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("nibabel is required to read NIfTI files.") from e
    img = nib.load(path)
    data = np.asarray(img.get_fdata())
    return img, data


def _peek_nifti_shape(path: str):
    """Read NIfTI header shape without loading full data (fast, for preflight)."""
    try:
        import nibabel as nib  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("nibabel is required to read NIfTI files.") from e
    img = nib.load(path)
    try:
        shp = img.header.get_data_shape()
    except Exception:
        shp = img.shape if hasattr(img, 'shape') else None
    return shp


def _l2norm_np(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (n + eps)


def _load_npy_float_matrix(path: str) -> np.ndarray:
    """Robustly load a float32 2D array from .npy.
    Accepts numeric arrays or pickled object arrays of lists/arrays, coercing to [K,D] float32.
    """
    try:
        arr = np.load(path, allow_pickle=False)
    except ValueError as e:
        msg = str(e)
        if "Object arrays cannot be loaded" in msg:
            arr = np.load(path, allow_pickle=True)
        else:
            raise
    # If object array of elements -> stack
    if arr.dtype == object:
        arr = [np.asarray(a, dtype=np.float32) for a in arr]
        arr = np.stack(arr, axis=0)
    else:
        arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array [K,D] in {path}, got shape {arr.shape}")
    return arr


def _maybe_encode_openclip(prompts_les: List[str], prompts_org: List[str], device: str,
                           model_name: str, pretrained: str):
    try:
        import torch
        import open_clip  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "open_clip is not available. Please install it or provide --t-les/--t-org .npy files."
        ) from e

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    with torch.no_grad():
        # Match BiomedCLIP trainer behavior: use context_length=256
        if str(model_name).lower().startswith("hf-hub:microsoft/biomedclip"):
            tok_les = tokenizer(prompts_les, context_length=256)
            tok_org = tokenizer(prompts_org, context_length=256)
        else:
            tok_les = tokenizer(prompts_les)
            tok_org = tokenizer(prompts_org)
        tles = model.encode_text(tok_les.to(device)).float()
        torg = model.encode_text(tok_org.to(device)).float()
        # L2 normalize
        tles = tles / (tles.norm(dim=-1, keepdim=True) + 1e-8)
        torg = torg / (torg.norm(dim=-1, keepdim=True) + 1e-8)
    return tles.cpu().numpy(), torg.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(
        description="Compute Sles/Sorg heatmaps from NIfTI features using fixed Prompt Bank."
    )
    parser.add_argument(
        "features",
        nargs="+",
        help=(
            "Paths to NIfTI feature volumes (4D) or directories containing them. "
            "Use --recursive to search subfolders."
        ),
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively search directories for .nii/.nii.gz")
    parser.add_argument("--out-dir", default="./promptbank_out", help="Output directory")
    parser.add_argument("--channels-first", action="store_true", help="Input features are [D,X,Y,Z] instead of [X,Y,Z,D]")
    parser.add_argument("--assume-normalized", action="store_true", help="Skip L2 normalization over channel D if features already normalized")
    parser.add_argument("--save-npy", action="store_true", help="Also save NPY heatmaps")

    # Text vectors
    parser.add_argument("--t-les", type=str, default=None, help="Path to pre-encoded lesion prompts .npy [K_les,D]")
    parser.add_argument("--t-org", type=str, default=None, help="Path to pre-encoded organ prompts .npy [K_org,D]")
    parser.add_argument(
        "--openclip",
        nargs=2,
        metavar=("MODEL", "PRETRAINED"),
        default=None,
        help=(
            "Encode built-in prompt bank with OpenCLIP (requires open_clip). "
            "If neither --t-les/--t-org nor --openclip are provided, defaults to "
            "BiomedCLIP: 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 open_clip_pytorch_model.bin'."
        ),
    )
    parser.add_argument("--device", default="cuda", help="Device for OpenCLIP (if used)")

    # Similarity/aggregation params
    parser.add_argument("--tau-les", type=float, default=0.07)
    parser.add_argument("--tau-org", type=float, default=0.07)
    parser.add_argument("--agg-les", choices=["mean", "lse"], default="lse")
    parser.add_argument("--agg-org", choices=["mean", "lse"], default="mean")
    parser.add_argument("--beta-les", type=float, default=10.0)
    parser.add_argument("--beta-org", type=float, default=10.0)
    parser.add_argument("--thr", type=float, default=0.5, help="Threshold for coverage/contrast metrics")
    # Optional gating and grid search
    parser.add_argument("--gate-lesion", action="store_true", help="Gate lesion map by organ mask (Sorg > gate-thr)")
    parser.add_argument("--gate-thr", type=float, default=None, help="Threshold for organ gating; defaults to --thr")
    parser.add_argument("--grid-search", action="store_true", help="Do small grid search over tau/beta")
    parser.add_argument("--grid-per-case", action="store_true", help="Run grid search per case (else only on first case)")
    parser.add_argument("--metric-key", choices=["corr", "cov_les", "cov_org", "top1p_les", "top1p_org", "les_in", "les_out"], default="corr", help="Grid search objective (minimizes median)")
    parser.add_argument("--tau-grid", type=str, default="0.03,0.05,0.07,0.1", help="Comma-separated tau candidates")
    parser.add_argument("--beta-grid", type=str, default="5,10,15", help="Comma-separated beta candidates")
    parser.add_argument("--skip-existing", action="store_true", help="Skip saving if output NIfTIs already exist")
    parser.add_argument("--z-chunk", type=int, default=0, help="Chunk size along Z to limit memory; 0 = no chunking")

    args = parser.parse_args()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    import torch

    # Prepare text vectors
    bank = build_prostate_prompt_bank()
    # Default to BiomedCLIP if no vectors nor model specified
    if args.t_les is None and args.t_org is None and args.openclip is None:
        args.openclip = (
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
            "open_clip_pytorch_model.bin",
        )

    if args.openclip is not None and not (args.t_les and args.t_org):
        model_name, pretrained = args.openclip
        tles_np, torg_np = _maybe_encode_openclip(bank["lesion"], bank["organ"], args.device, model_name, pretrained)
    else:
        if args.t_les is None or args.t_org is None:
            raise SystemExit("Provide --t-les and --t-org .npy files, or use --openclip to encode prompts.")
        tles_np = _load_npy_float_matrix(args.t_les)
        torg_np = _load_npy_float_matrix(args.t_org)

    # Expand inputs: allow files and/or directories
    def _expand_inputs(paths, recursive=False):
        out = []
        for p in paths:
            if os.path.isdir(p):
                if recursive:
                    for root, _, files in os.walk(p):
                        for fn in files:
                            if fn.endswith(".nii") or fn.endswith(".nii.gz"):
                                out.append(os.path.join(root, fn))
                else:
                    for fn in os.listdir(p):
                        if fn.endswith(".nii") or fn.endswith(".nii.gz"):
                            out.append(os.path.join(p, fn))
            else:
                out.append(p)
        # de-dup and sort for determinism
        out = sorted(dict.fromkeys(out))
        return out

    feature_paths = _expand_inputs(args.features, recursive=args.recursive)
    if not feature_paths:
        raise SystemExit("No NIfTI feature files found. Check your path or use --recursive.")

    # Build evaluator (no text encoder; pass pre-encoded vectors)
    evaluator = PromptBankEvaluator(text_encoder=None, prompt_bank=bank, device=args.device)

    def _clean_name(p: str) -> str:
        base = os.path.basename(p)
        if base.endswith(".nii.gz"):
            return base[:-7]
        if base.endswith(".nii"):
            return base[:-4]
        return os.path.splitext(base)[0]

    # Early D-dim check on first file to fail fast (use header only, fast)
    first_path = feature_paths[0]
    print(f"Preflight: reading header to determine feature dim -> {first_path}", flush=True)
    shp = _peek_nifti_shape(first_path)
    if shp is None or len(shp) != 4:
        raise SystemExit(f"Feature volume must be 4D [X,Y,Z,D], got shape {shp} for {first_path}")
    D = int(shp[0] if args.channels_first else shp[-1])
    # If text dim mismatches feature dim, mirror trainer's fallback: project with orthogonal matrix
    if tles_np.shape[1] != D or torg_np.shape[1] != D:
        import torch
        print(
            f"WARNING: Channel dim mismatch: features D={D}, T_les D={tles_np.shape[1]}, T_org D={torg_np.shape[1]}; projecting text to {D}.",
            flush=True,
        )
        tdim = int(tles_np.shape[1])
        if int(torg_np.shape[1]) != tdim:
            raise SystemExit(f"Lesion/organ text dims differ: {tles_np.shape[1]} vs {torg_np.shape[1]}")
        import numpy as _np
        w = torch.empty((tdim, D), dtype=torch.float32)
        torch.manual_seed(0)
        torch.nn.init.orthogonal_(w)
        tles_base = _np.asarray(tles_np, dtype=_np.float32)
        torg_base = _np.asarray(torg_np, dtype=_np.float32)
        Tles_proj = torch.tensor(tles_base, dtype=torch.float32) @ w  # [K,D]
        Torg_proj = torch.tensor(torg_base, dtype=torch.float32) @ w
        tles_np = Tles_proj.cpu().numpy().astype(_np.float32)
        torg_np = Torg_proj.cpu().numpy().astype(_np.float32)

    # Now that dims are aligned, move text vectors to device
    T_les = torch.tensor(np.ascontiguousarray(tles_np, dtype=np.float32), dtype=torch.float32, device=evaluator.device)
    T_org = torch.tensor(np.ascontiguousarray(torg_np, dtype=np.float32), dtype=torch.float32, device=evaluator.device)

    # Process each file independently to support heterogeneous spatial shapes
    total = len(feature_paths)
    print(f"Found {total} feature file(s). Processing sequentially...", flush=True)
    # Prepare grid candidates
    def _parse_floats(s: str) -> Tuple[float, ...]:
        return tuple(float(x) for x in s.split(',') if x.strip())
    tau_candidates = _parse_floats(args.tau_grid)
    beta_candidates = _parse_floats(args.beta_grid)
    gate_thr = args.thr if args.gate_thr is None else args.gate_thr

    # Optionally determine best params on the first case
    best_global = None
    for idx, p in enumerate(feature_paths, 1):
        try:
            print(f"[{idx}/{total}] Loading features: {p}", flush=True)
            ref_img, data = _load_nifti(p)
            if args.channels_first:
                if data.ndim != 4:
                    raise SystemExit(f"Feature volume must be 4D, got shape {data.shape} for {p}")
                data = np.moveaxis(data, 0, -1)  # [D,X,Y,Z] -> [X,Y,Z,D]
            if data.ndim != 4:
                raise SystemExit(f"Feature volume must be 4D [X,Y,Z,D], got {data.shape} for {p}")
            if not args.assume_normalized:
                data = _l2norm_np(data, axis=-1)
            # Be robust to array subclasses/non-contiguous arrays from nibabel
            data_np = np.ascontiguousarray(np.array(data, dtype=np.float32, copy=False))
            X, Y, Z, _Dd = data_np.shape
            if args.z_chunk and args.z_chunk > 0:
                # Defer tensor creation per-chunk to limit GPU memory
                feats_t = None
            else:
                feats_t = (
                    torch.tensor(data_np, dtype=torch.float32)
                    .permute(3, 0, 1, 2)
                    .unsqueeze(0)
                    .contiguous()
                    .to(evaluator.device, non_blocking=True)
                )

            # Optionally grid search on this case (or only on first case)
            if args.grid_search and (args.grid_per_case or (best_global is None)):
                from experiments.promptbank_eval_prostate_test import grid_search_params  # type: ignore
                # When chunking, build a small central slab for grid search to avoid peak memory
                if feats_t is None:
                    zc = int(args.z_chunk) if (args.z_chunk and args.z_chunk > 0) else max(1, Z // 8)
                    zc = max(1, min(zc, Z))
                    z0 = max(0, (Z - zc) // 2)
                    z1 = min(Z, z0 + zc)
                    sub = data_np[:, :, z0:z1, :]
                    feats_for_gs = (
                        torch.tensor(sub, dtype=torch.float32)
                        .permute(3, 0, 1, 2)
                        .unsqueeze(0)
                        .contiguous()
                        .to(evaluator.device, non_blocking=True)
                    )
                    slab_info = f"[Z {z0}:{z1}]"
                else:
                    feats_for_gs = feats_t
                    slab_info = "[full]"

                best = grid_search_params(
                    feats_for_gs,
                    T_les=T_les,
                    T_org=T_org,
                    tau_grid=tau_candidates,
                    beta_grid=beta_candidates,
                    agg_les=args.agg_les,
                    agg_org=args.agg_org,
                    metric_key=args.metric_key,
                )
                # Free temporary grid-search tensor if chunked
                if feats_t is None:
                    del feats_for_gs
                    torch.cuda.empty_cache() if evaluator.device.type == 'cuda' else None
                print(f"[{idx}/{total}] Grid best -> tau={best['tau']} beta={best['beta']} score={best['score']:.4f} ({args.metric_key})", flush=True)
                if not args.grid_per_case:
                    best_global = best

            # Select params
            if best_global is not None:
                tau_les = tau_org = float(best_global['tau'])
                beta_les = beta_org = float(best_global['beta'])
            elif args.grid_search and 'best' in locals():
                tau_les = tau_org = float(best['tau'])
                beta_les = beta_org = float(best['beta'])
            else:
                tau_les, tau_org = args.tau_les, args.tau_org
                beta_les, beta_org = args.beta_les, args.beta_org

            # Compute heatmaps for this case
            if feats_t is not None:
                Sles, Sorg = evaluator.compute_heatmaps(
                    feats_t,
                    T_les=T_les,
                    T_org=T_org,
                    tau_les=tau_les,
                    tau_org=tau_org,
                    agg_les=args.agg_les,
                    agg_org=args.agg_org,
                    beta_les=beta_les,
                    beta_org=beta_org,
                    normalize=True,
                )
            else:
                # Chunked along Z
                sles_np = np.zeros((1, X, Y, Z), dtype=np.float32)
                sorg_np = np.zeros((1, X, Y, Z), dtype=np.float32)
                for z0 in range(0, Z, int(args.z_chunk)):
                    z1 = min(Z, z0 + int(args.z_chunk))
                    sub = data_np[:, :, z0:z1, :]  # [X,Y,zc,D]
                    feats_chunk = (
                        torch.tensor(sub, dtype=torch.float32)
                        .permute(3, 0, 1, 2)
                        .unsqueeze(0)
                        .contiguous()
                        .to(evaluator.device, non_blocking=True)
                    )
                    Sles_c, Sorg_c = evaluator.compute_heatmaps(
                        feats_chunk,
                        T_les=T_les,
                        T_org=T_org,
                        tau_les=tau_les,
                        tau_org=tau_org,
                        agg_les=args.agg_les,
                        agg_org=args.agg_org,
                        beta_les=beta_les,
                        beta_org=beta_org,
                        normalize=True,
                    )
                    sles_np[0, :, :, z0:z1] = Sles_c.detach().cpu().numpy()[0]
                    sorg_np[0, :, :, z0:z1] = Sorg_c.detach().cpu().numpy()[0]
                    # Free chunk tensors
                    del feats_chunk, Sles_c, Sorg_c
                    torch.cuda.empty_cache() if evaluator.device.type == 'cuda' else None
                Sles = torch.tensor(sles_np, dtype=torch.float32)
                Sorg = torch.tensor(sorg_np, dtype=torch.float32)

            # Metrics and save
            m = evaluator.separability_metrics(Sles, Sorg, thr=args.thr)[0]
            # Optional gating by organ mask
            if args.gate_lesion:
                Rorg = (Sorg > gate_thr).float()
                Sles_g = Sles * Rorg
                # For fair comparison, use the same threshold as gating when reporting mg
                mg = evaluator.separability_metrics(Sles_g, Sorg, thr=gate_thr)[0]
                m_print = (
                    f"corr={m['corr']:.3f}->{mg['corr']:.3f} cov_les={m['cov_les']:.3f}->{mg['cov_les']:.3f} "
                    f"cov_org={m['cov_org']:.3f} top1p_les={m['top1p_les']:.3f}->{mg['top1p_les']:.3f} "
                    f"les_in={m['les_in']:.3f}->{mg['les_in']:.3f} les_out={m['les_out']:.3f}->{mg['les_out']:.3f}"
                )
                Sles = Sles_g
            else:
                m_print = (
                    f"corr={m['corr']:.3f} cov_les={m['cov_les']:.3f} cov_org={m['cov_org']:.3f} "
                    f"top1p_les={m['top1p_les']:.3f} top1p_org={m['top1p_org']:.3f} les_in={m['les_in']:.3f} les_out={m['les_out']:.3f}"
                )
            nm = _clean_name(p)
            print(f"[{idx}/{total}] {nm}: {m_print}", flush=True)

            # Skip saving if outputs exist
            if args.skip_existing:
                from pathlib import Path as _P
                les_path = _P(out_dir) / f"Sles_{nm}.nii.gz"
                org_path = _P(out_dir) / f"Sorg_{nm}.nii.gz"
                if les_path.exists() and org_path.exists():
                    print(f"[{idx}/{total}] Skipping save (exists)", flush=True)
                else:
                    save_heatmaps_nifti(Sles, Sorg, ref_nii_paths=[p], out_dir=out_dir, names=[nm])
            else:
                save_heatmaps_nifti(Sles, Sorg, ref_nii_paths=[p], out_dir=out_dir, names=[nm])
            if args.save_npy:
                save_heatmaps_npy(Sles, Sorg, out_dir=out_dir, names=[nm])
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"[{idx}/{total}] ERROR processing {p}: {e}\n{tb}", flush=True)
            continue

    print(f"Done. Saved heatmaps to: {out_dir}")


if __name__ == "__main__":
    main()
