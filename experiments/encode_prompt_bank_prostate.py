"""
Encode the fixed prostate Prompt Bank (lesion & organ) into text vectors and
save as .npy files (T_les.npy, T_org.npy). Use either:

- Your own text encoder function imported from a module:
  --encoder-module mypkg.myencoder --encoder-func encode_text
  where the function signature is: encode_text(List[str]) -> torch.FloatTensor [K, D]

- Or OpenCLIP (if installed):
  --openclip MODEL PRETRAINED

Outputs are L2-normalized over the embedding dimension.
"""

import argparse
import os
from typing import List, Tuple

# Self-contained prompt bank to avoid importing torch-heavy modules when only
# encoding prompts. Matches your specified phrases exactly.
def build_prostate_prompt_bank() -> dict:
    lesion = [
        "lesion",
        "tumor",
        "suspicious focus",
        "low ADC",
        "DWI hyperintense",
        "T2 hypointense nodule",
    ]
    organ = [
        "prostate",
        "peripheral zone",
        "transition zone",
        "central zone",
        "anterior fibromuscular stroma",
        "prostatic capsule",
    ]
    return {"lesion": lesion, "organ": organ}

import numpy as np


def _l2norm_np(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (n + eps)


def _encode_openclip(prompts: List[str], device: str, model_name: str, pretrained: str) -> np.ndarray:
    try:
        import torch
        import open_clip  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("open_clip is required for --openclip mode.") from e

    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(model_name)
    with torch.no_grad():
        # BiomedCLIP expects a longer context length (256)
        if str(model_name).lower().startswith("hf-hub:microsoft/biomedclip"):
            tokens = tokenizer(prompts, context_length=256)
        else:
            tokens = tokenizer(prompts)
        emb = model.encode_text(tokens.to(device)).float()
        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
    return emb.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Encode fixed prostate Prompt Bank to .npy vectors")
    parser.add_argument("--out-dir", default="./promptbank_vectors", help="Directory to save T_les.npy/T_org.npy")

    # Use custom encoder
    parser.add_argument("--encoder-module", type=str, default=None, help="Module path containing the encoder function")
    parser.add_argument("--encoder-func", type=str, default="encode_text", help="Function name inside the module")
    parser.add_argument("--device", default="cuda", help="Device for custom encoder/OpenCLIP")

    # Or use OpenCLIP
    parser.add_argument(
        "--openclip",
        nargs=2,
        metavar=("MODEL", "PRETRAINED"),
        default=None,
        help=(
            "Encode prompts with OpenCLIP. If neither --encoder-module nor --openclip "
            "is provided, defaults to BiomedCLIP: 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 open_clip_pytorch_model.bin'."
        ),
    )

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    bank = build_prostate_prompt_bank()
    prompts_les = bank["lesion"]
    prompts_org = bank["organ"]

    # Default to BiomedCLIP if no encoder specified
    if args.encoder_module is None and args.openclip is None:
        args.openclip = (
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
            "open_clip_pytorch_model.bin",
        )

    # Encode
    if args.openclip is not None and args.encoder_module is None:
        model_name, pretrained = args.openclip
        T_les = _encode_openclip(prompts_les, args.device, model_name, pretrained)
        T_org = _encode_openclip(prompts_org, args.device, model_name, pretrained)
    else:
        if args.encoder_module is None:
            raise SystemExit("Provide --encoder-module (and optionally --encoder-func), or use --openclip.")
        import importlib
        import torch

        mod = importlib.import_module(args.encoder_module)
        if not hasattr(mod, args.encoder_func):
            raise SystemExit(f"Function {args.encoder_func} not found in module {args.encoder_module}")
        encode_text = getattr(mod, args.encoder_func)
        with torch.no_grad():
            tles_t = encode_text(prompts_les)
            torg_t = encode_text(prompts_org)
            if not isinstance(tles_t, torch.Tensor) or not isinstance(torg_t, torch.Tensor):
                raise SystemExit("Encoder must return torch.Tensor")
            tles_t = tles_t.to(args.device).float()
            torg_t = torg_t.to(args.device).float()
            # L2 normalize
            tles_t = tles_t / (tles_t.norm(dim=-1, keepdim=True) + 1e-8)
            torg_t = torg_t / (torg_t.norm(dim=-1, keepdim=True) + 1e-8)
        T_les = tles_t.cpu().numpy()
        T_org = torg_t.cpu().numpy()

    # Save vectors and prompts for transparency
    les_path = os.path.join(args.out_dir, "T_les.npy")
    org_path = os.path.join(args.out_dir, "T_org.npy")
    np.save(les_path, T_les.astype(np.float32))
    np.save(org_path, T_org.astype(np.float32))

    with open(os.path.join(args.out_dir, "prompts_les.txt"), "w") as f:
        for s in prompts_les:
            f.write(s + "\n")
    with open(os.path.join(args.out_dir, "prompts_org.txt"), "w") as f:
        for s in prompts_org:
            f.write(s + "\n")

    # Report shapes
    print(f"Saved: {les_path} shape={T_les.shape}")
    print(f"Saved: {org_path} shape={T_org.shape}")


if __name__ == "__main__":
    main()
