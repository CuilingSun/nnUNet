"""
Extract A1-projected 4D feature volumes from raw NIfTI images using
BiomedCLIP's ViT patch token embeddings projected into the text space.

For each axial slice:
- Robustly scale intensities per volume (percentile 1–99) and clip to [0,1].
- Convert to RGB and preprocess to 224x224 using OpenCLIP's transforms.
- Run BiomedCLIP visual encoder and extract per-patch token embeddings.
- Apply the visual projection to map tokens to the joint embed dim (text space).
- Bilinearly upsample token grid back to original slice resolution.
- Stack slices to form a 4D feature volume [X, Y, Z, D], L2-normalized over D.

Outputs: NIfTI files with suffix "_featA1.nii.gz" stored under --out-dir.

Notes:
- This is a pragmatic spatialization of CLIP features; it approximates an A1 head
  by applying the visual projection to each patch token.
- Requires: torch, nibabel, open_clip_torch, Pillow, numpy.
"""

import argparse
import os
from typing import List, Tuple

import numpy as np


def _load_nifti(path: str):
    try:
        import nibabel as nib  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("nibabel is required to read NIfTI files.") from e
    img = nib.load(path)
    data = np.asarray(img.get_fdata())
    return img, data


def _save_nifti_like(ref_img, data: np.ndarray, out_path: str):
    import nibabel as nib  # type: ignore
    img = nib.Nifti1Image(data.astype(np.float32), ref_img.affine, ref_img.header)
    nib.save(img, out_path)


def _robust_scale(vol: np.ndarray, p_low=1.0, p_high=99.0) -> np.ndarray:
    x = vol.astype(np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.float32)
    lo = np.percentile(x[finite], p_low)
    hi = np.percentile(x[finite], p_high)
    if hi <= lo:
        hi = lo + 1.0
    x = (x - lo) / (hi - lo)
    x = np.clip(x, 0.0, 1.0)
    return x


def _to_rgb_pil(slice01: np.ndarray):
    # slice01: [H, W] in [0,1]
    from PIL import Image
    arr = (slice01 * 255.0).astype(np.uint8)
    img = Image.fromarray(arr, mode="L").convert("RGB")
    return img


def _extract_patch_tokens_visual(model_visual, x):
    """
    Try multiple ViT variants to obtain per-patch tokens projected to the
    joint embedding dimension. Supports OpenCLIP ViT and timm-like ViT.

    Args:
        model_visual: visual backbone from OpenCLIP
        x: [B,3,224,224]
    Returns:
        tokens_proj: [B, N, D], grid_h, grid_w
    """
    import torch

    B = x.shape[0]

    # Helper: recursively locate a ViT-like module exposing conv1 or patch_embed
    def _find_vit_like(m, visited=None):
        if visited is None:
            visited = set()
        if id(m) in visited:
            return None
        visited.add(id(m))
        if hasattr(m, "conv1") and hasattr(m, "class_embedding"):
            return m
        if hasattr(m, "patch_embed") and hasattr(m, "blocks"):
            return m
        for attr in ("trunk", "model", "visual"):
            if hasattr(m, attr):
                sub = getattr(m, attr)
                res = _find_vit_like(sub, visited)
                if res is not None:
                    return res
        # search children
        for child in getattr(m, "children", lambda: [])():
            res = _find_vit_like(child, visited)
            if res is not None:
                return res
        return None

    vit = _find_vit_like(model_visual)
    if vit is None:
        raise AssertionError("Unsupported visual backbone; expected ViT-style with conv1 or patch_embed")

    # Case 1: OpenCLIP ViT with conv1/class_embedding/etc.
    if hasattr(vit, "conv1") and hasattr(vit, "class_embedding"):
        dtype = model_visual.conv1.weight.dtype
        x = x.to(dtype)
        feats = vit.conv1(x)  # [B, width, gh, gw]
        gh, gw = feats.shape[-2], feats.shape[-1]
        feats = feats.reshape(B, feats.shape[1], gh * gw).permute(0, 2, 1)  # [B, N, width]
        # add cls token + pos emb
        cls_tok = vit.class_embedding.to(feats.dtype).unsqueeze(0).expand(B, -1, -1)
        feats = torch.cat([cls_tok, feats], dim=1)  # [B, 1+N, width]
        pos = vit.positional_embedding.to(feats.dtype)
        if pos.shape[0] != feats.shape[1]:
            if pos.shape[0] < feats.shape[1]:
                pad = feats.shape[1] - pos.shape[0]
                pos = torch.cat([pos, pos[-pad:]], dim=0)
            else:
                pos = pos[: feats.shape[1]]
        feats = feats + pos
        feats = vit.ln_pre(feats)
        feats = feats.permute(1, 0, 2)
        feats = vit.transformer(feats)
        feats = feats.permute(1, 0, 2)
        tokens = feats[:, 1:, :]
        # post norm and projection
        if hasattr(vit, "ln_post"):
            tokens = vit.ln_post(tokens)
        proj = getattr(vit, "proj", None)
        if proj is not None:
            tokens = tokens @ proj
        return tokens, gh, gw

    # Case 2: timm-like ViT (patch_embed/blocks/norm)
    if hasattr(vit, "patch_embed") and hasattr(vit, "blocks"):
        v = vit
        # dtype from first block param or patch_embed
        for m in [getattr(v.patch_embed, 'proj', None), v]:
            if m is not None:
                try:
                    dtype = next(p for p in m.parameters()).dtype
                    break
                except StopIteration:
                    continue
        else:
            dtype = x.dtype
        x = x.to(dtype)
        feats = v.patch_embed(x)  # [B, N, C] or [B, C, gh, gw]
        if feats.dim() == 4:
            gh, gw = feats.shape[-2], feats.shape[-1]
            feats = feats.flatten(2).transpose(1, 2)  # [B, N, C]
        else:
            # try to recover grid from attribute
            grid = getattr(v.patch_embed, 'grid_size', None)
            if grid is not None:
                gh, gw = int(grid[0]), int(grid[1])
            else:
                N = feats.shape[1]
                gh = gw = int(N ** 0.5)
        # prepend cls token if present
        if hasattr(v, "cls_token") and v.cls_token is not None:
            cls = v.cls_token.expand(feats.shape[0], -1, -1)
            feats = torch.cat((cls, feats), dim=1)
        # add pos embed if present
        if hasattr(v, "pos_embed") and v.pos_embed is not None:
            pos = v.pos_embed
            if pos.shape[1] != feats.shape[1]:
                # naive adjust: tile/crop
                if pos.shape[1] < feats.shape[1]:
                    pad = feats.shape[1] - pos.shape[1]
                    pos = torch.cat([pos, pos[:, -pad:, :]], dim=1)
                else:
                    pos = pos[:, :feats.shape[1], :]
            feats = feats + pos
        # pos drop / norm pre
        if hasattr(v, "pos_drop"):
            feats = v.pos_drop(feats)
        # transformer blocks
        for blk in v.blocks:
            feats = blk(feats)
        # norm
        if hasattr(v, "norm") and v.norm is not None:
            feats = v.norm(feats)
        # remove cls token if present
        if hasattr(v, "cls_token") and v.cls_token is not None and feats.shape[1] == gh * gw + 1:
            tokens = feats[:, 1:, :]
        else:
            tokens = feats
        # project to embed dim if projection exists
        proj = getattr(v, "proj", None)
        if proj is not None:
            tokens = tokens @ proj
        return tokens, gh, gw

    raise AssertionError("Unsupported visual backbone; expected ViT-style with conv1 or patch_embed")


def main():
    parser = argparse.ArgumentParser(description="Extract A1-like 4D feature volumes using BiomedCLIP")
    parser.add_argument("inputs", nargs="+", help="NIfTI images or folders (use --recursive for subfolders)")
    parser.add_argument("--recursive", action="store_true", help="Recursively search folders for .nii/.nii.gz")
    parser.add_argument("--out-dir", default="./features_a1_biomedclip", help="Output directory")
    parser.add_argument("--device", default="cuda", help="Torch device")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for slice processing")
    parser.add_argument("--fp16", action="store_true", help="Use float16 in model for speed/memory (saves as float32)")
    parser.add_argument("--limit-files", type=int, default=0, help="Process only the first N files (0 = all)")
    parser.add_argument("--max-slices", type=int, default=0, help="Process only the first K axial slices per volume (0 = all)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip files whose output already exists")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Expand inputs
    def _expand(paths, recursive=False):
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
        return sorted(dict.fromkeys(out))

    files = _expand(args.inputs, recursive=args.recursive)
    if not files:
        raise SystemExit("No NIfTI files found.")
    if args.limit_files and args.limit_files > 0:
        files = files[: args.limit_files]
    print(f"Found {len(files)} file(s). Output dir: {args.out_dir}", flush=True)
    for show in files[:3]:
        print(f"  - {show}", flush=True)

    # Load BiomedCLIP via OpenCLIP
    try:
        import torch
        import torch.nn.functional as F
        import open_clip  # type: ignore
    except Exception as e:
        raise RuntimeError("Requires torch and open_clip_torch installed.") from e

    model_name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    pretrained = "open_clip_pytorch_model.bin"
    print("Loading BiomedCLIP (may download on first run)...", flush=True)
    model, _, _ = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=args.device
    )
    model.eval()
    if args.fp16:
        model = model.half()
    print("Model loaded.", flush=True)

    # Manual CLIP preprocessing to avoid torchvision to_tensor issues
    from PIL import Image
    import torch

    _CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    _CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

    def _prep_clip(pil_img: Image.Image, size: int = 224) -> torch.Tensor:
        # Resize with bicubic, convert to float tensor normalized by CLIP stats
        pil_resized = pil_img.resize((size, size), resample=Image.BICUBIC)
        arr = np.asarray(pil_resized, dtype=np.float32) / 255.0  # [H,W,3]
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        arr = (arr - _CLIP_MEAN) / _CLIP_STD
        arr = np.transpose(arr, (2, 0, 1))  # [3, H, W]
        # Use torch.tensor to avoid from_numpy type issues across environments
        return torch.tensor(arr, dtype=torch.float32)

    def _stem_from_path(p: str) -> str:
        base = os.path.basename(p)
        if base.endswith(".nii.gz"):
            return base[:-7]
        if base.endswith(".nii"):
            return base[:-4]
        return os.path.splitext(base)[0]

    total = len(files)
    for idx, p in enumerate(files, 1):
        stem = _stem_from_path(p)
        out_path = os.path.join(args.out_dir, f"{stem}_featA1.nii.gz")
        if args.skip_existing and os.path.isfile(out_path):
            print(f"[{idx}/{total}] Skip (exists): {p} -> {out_path}", flush=True)
            continue
        print(f"[{idx}/{total}] Processing: {p}", flush=True)
        ref_img, vol = _load_nifti(p)
        if vol.ndim != 3:
            print(f"Skip non-3D file: {p} shape={vol.shape}")
            continue

        vol01 = _robust_scale(vol)
        X, Y, Z = vol01.shape
        # Prepare output feature volume [X,Y,Z,D]
        # We defer allocating until we know D
        feat_vol = None

        # Process slices in batches for speed
        zs = list(range(Z))
        if args.max_slices and args.max_slices > 0:
            zs = zs[: args.max_slices]
        for i in range(0, len(zs), args.batch_size):
            batch_idx = zs[i : i + args.batch_size]
            imgs = []
            for z in batch_idx:
                pil = _to_rgb_pil(vol01[:, :, z])
                imgs.append(_prep_clip(pil))  # tensor [3,224,224] normalized
            x = torch.stack(imgs, dim=0).to(args.device, non_blocking=True)
            if args.fp16:
                x = x.half()

            with torch.no_grad():
                tokens, gh, gw = _extract_patch_tokens_visual(model.visual, x)
                # tokens: [B, N, D] -> [B, D, gh, gw]
                B = tokens.shape[0]
                D = tokens.shape[-1]
                tokens_grid = tokens.reshape(B, gh, gw, D).permute(0, 3, 1, 2).contiguous()
                # Upsample to [B, D, X, Y]
                up = F.interpolate(tokens_grid, size=(X, Y), mode="bilinear", align_corners=False)
                # L2 normalize per spatial location over D
                up = up / (up.norm(dim=1, keepdim=True) + 1e-8)

            up_np = up.detach().float().cpu().numpy()  # [B, D, X, Y]
            # Initialize feature volume on first batch
            if feat_vol is None:
                feat_vol = np.zeros((X, Y, Z, D), dtype=np.float32)

            for j, z in enumerate(batch_idx):
                # [D,X,Y] -> [X,Y,D]
                feat_vol[:, :, z, :] = np.transpose(up_np[j], (1, 2, 0))

        # Save output NIfTI
        # Save output NIfTI
        _save_nifti_like(ref_img, feat_vol, out_path)
        print(f"Saved features: {out_path} shape={feat_vol.shape}", flush=True)


if __name__ == "__main__":
    main()
