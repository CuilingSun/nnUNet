"""
Standalone evaluator for fixed Prompt Bank (lesion & organ) for prostate.

What's inside:
- build_prostate_prompt_bank(): returns your specified English phrases.
- PromptBankEvaluator: encode prompts, compute Sles/Sorg heatmaps, separability metrics.
- save_heatmaps_npy/save_heatmaps_nifti: utilities to persist heatmaps per case.

This module does not import or modify your existing project code. You can import
it from your own scripts and pass in your text encoder and A1-projected features.
"""

from typing import List, Dict, Tuple, Optional, Callable, Sequence
import os
import numpy as np
import torch


def build_prostate_prompt_bank() -> Dict[str, List[str]]:
    """
    Uses exactly the phrases you specified (English).

    Returns a dict with keys 'lesion' and 'organ'.
    """
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


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def _minmax_per_image(S: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Min-max normalize each image independently to [0,1].
    S: [B, *spatial]
    """
    B = S.shape[0]
    Sf = S.view(B, -1)
    smin = Sf.min(dim=1, keepdim=True).values
    smax = Sf.max(dim=1, keepdim=True).values
    Sn = (Sf - smin) / (smax - smin + eps)
    return Sn.view_as(S)


class PromptBankEvaluator:
    def __init__(
        self,
        text_encoder: Optional[Callable[[List[str]], torch.Tensor]],
        prompt_bank: Dict[str, List[str]],
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        """
        text_encoder: callable(List[str]) -> torch.FloatTensor [K, D_text] (unnormalized).
                      If None, you must pass pre-encoded T_les/T_org to compute_heatmaps.
        prompt_bank: dict with keys 'lesion' and 'organ'.
        """
        self.text_encoder = text_encoder
        self.prompt_bank = prompt_bank
        self.device = (
            torch.device(device)
            if (device != "cuda" or torch.cuda.is_available())
            else torch.device("cpu")
        )
        self.dtype = dtype
        self.T_les: Optional[torch.Tensor] = None
        self.T_org: Optional[torch.Tensor] = None

    @torch.no_grad()
    def encode_bank(self) -> Tuple[torch.Tensor, torch.Tensor]:
        assert (
            self.text_encoder is not None
        ), "text_encoder is None; pass T_les/T_org directly instead."
        les = self.prompt_bank["lesion"]
        org = self.prompt_bank["organ"]
        T_les = self.text_encoder(les).to(self.device, self.dtype)
        T_org = self.text_encoder(org).to(self.device, self.dtype)
        T_les = _l2norm(T_les, dim=-1)
        T_org = _l2norm(T_org, dim=-1)
        self.T_les, self.T_org = T_les, T_org
        return T_les, T_org

    @torch.no_grad()
    def compute_heatmaps(
        self,
        feats: torch.Tensor,
        T_les: Optional[torch.Tensor] = None,
        T_org: Optional[torch.Tensor] = None,
        tau_les: float = 0.07,
        tau_org: float = 0.07,
        agg_les: str = "lse",
        agg_org: str = "mean",
        beta_les: float = 10.0,
        beta_org: float = 10.0,
        normalize: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        feats: [B, D, *spatial], already projected to text space and L2-normalized along D.
        Returns:
            Sles, Sorg: [B, *spatial], in [0,1] if normalize=True
        """
        assert feats.dim() >= 3, "feats must be [B, D, *spatial]"
        B, D = feats.shape[:2]
        spatial_shape = feats.shape[2:]
        N = int(torch.tensor(spatial_shape).prod().item())
        X = feats.reshape(B, D, N).transpose(1, 2).contiguous()  # [B, N, D]

        Tles = T_les if T_les is not None else self.T_les
        Torg = T_org if T_org is not None else self.T_org
        assert Tles is not None and Torg is not None, "Encode prompts first or pass T_les/T_org"
        assert Tles.shape[1] == D and Torg.shape[1] == D, "Text dim must match feature dim"

        def _sims(X: torch.Tensor, T: torch.Tensor, tau: float) -> torch.Tensor:
            # X: [B, N, D], T: [K, D] -> [B, N, K]
            return torch.einsum("bnd,kd->bnk", X, T) / tau

        def _agg(S: torch.Tensor, how: str, beta: float) -> torch.Tensor:
            if how == "mean":
                return S.mean(dim=-1)  # [B, N]
            if how == "lse":
                return (1.0 / beta) * torch.logsumexp(beta * S, dim=-1)
            raise ValueError("agg must be 'mean' or 'lse'")

        Sles = _agg(_sims(X, Tles, tau_les), agg_les, beta_les).reshape(B, *spatial_shape)
        Sorg = _agg(_sims(X, Torg, tau_org), agg_org, beta_org).reshape(B, *spatial_shape)

        if normalize:
            Sles = _minmax_per_image(Sles)
            Sorg = _minmax_per_image(Sorg)
        return Sles, Sorg

    @torch.no_grad()
    def separability_metrics(
        self, Sles: torch.Tensor, Sorg: torch.Tensor, thr: float = 0.5
    ) -> List[Dict[str, float]]:
        """
        Sles, Sorg: [B, *spatial], ideally in [0,1].
        Returns a list of dicts with quick separability indicators per image.
        """
        assert Sles.shape == Sorg.shape, "Sles/Sorg shapes must match"
        B = Sles.shape[0]
        Sles_f = Sles.view(B, -1)
        Sorg_f = Sorg.view(B, -1)

        def _safe_quantile(x: torch.Tensor, q: float, max_elems: int = 10_000_000) -> torch.Tensor:
            """
            Robust quantile for very large 1D tensors. If x is too large, subsample
            evenly to at most max_elems before calling torch.quantile.
            """
            x = x.float().view(-1)
            n = x.numel()
            if n == 0:
                return torch.tensor(float('nan'), device=x.device, dtype=x.dtype)
            if n > max_elems:
                step = int((n + max_elems - 1) // max_elems)
                x = x[::step]
            return torch.quantile(x, q)
        out = []
        for i in range(B):
            a = Sles_f[i]
            b = Sorg_f[i]
            # Pearson correlation
            ab = torch.stack([a, b], dim=0)
            corr = torch.corrcoef(ab)[0, 1].item()
            # Coverage
            cov_les = (a > thr).float().mean().item()
            cov_org = (b > thr).float().mean().item()

            # Top 1% mass ratio
            def top_ratio(x: torch.Tensor, q: float = 0.99) -> float:
                t = _safe_quantile(x, q)
                if not torch.isfinite(t):
                    return float('nan')
                s_all = x.sum()
                if s_all.abs().item() == 0.0:
                    return float('nan')
                return (x[x >= t].sum() / (s_all + 1e-8)).item()

            tr_les = top_ratio(a)
            tr_org = top_ratio(b)

            # Lesion contrast inside vs outside organ
            Rorg = b > thr
            m_in = a[Rorg].mean().item() if Rorg.any() else float("nan")
            m_out = a[~Rorg].mean().item() if (~Rorg).any() else float("nan")

            out.append(
                dict(
                    corr=corr,
                    cov_les=cov_les,
                    cov_org=cov_org,
                    top1p_les=tr_les,
                    top1p_org=tr_org,
                    les_in=m_in,
                    les_out=m_out,
                )
            )
        return out


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_heatmaps_npy(
    Sles: torch.Tensor,
    Sorg: torch.Tensor,
    out_dir: str,
    names: Optional[Sequence[str]] = None,
    prefix_les: str = "Sles",
    prefix_org: str = "Sorg",
) -> List[Tuple[str, str]]:
    """
    Save heatmaps per case as .npy files.

    Sles, Sorg: [B, *spatial] torch tensors.
    names: optional case identifiers; default uses zero-based indices.
    Returns list of (les_path, org_path) for each case.
    """
    assert Sles.shape == Sorg.shape, "Sles/Sorg shapes must match"
    _ensure_dir(out_dir)
    B = Sles.shape[0]
    if names is None:
        names = [f"{i:04d}" for i in range(B)]
    assert len(names) == B, "names length must match batch size"

    out_paths: List[Tuple[str, str]] = []
    Sles_np = Sles.detach().cpu().numpy()
    Sorg_np = Sorg.detach().cpu().numpy()
    for i in range(B):
        les_path = os.path.join(out_dir, f"{prefix_les}_{names[i]}.npy")
        org_path = os.path.join(out_dir, f"{prefix_org}_{names[i]}.npy")
        np.save(les_path, Sles_np[i])
        np.save(org_path, Sorg_np[i])
        out_paths.append((les_path, org_path))
    return out_paths


def save_heatmaps_nifti(
    Sles: torch.Tensor,
    Sorg: torch.Tensor,
    ref_nii_paths: Sequence[str],
    out_dir: str,
    names: Optional[Sequence[str]] = None,
    prefix_les: str = "Sles",
    prefix_org: str = "Sorg",
) -> List[Tuple[str, str]]:
    """
    Save heatmaps per case as NIfTI using reference images to copy affine/header.

    Requirements: nibabel must be installed and Sles/Sorg spatial shapes must
    match the reference NIfTI data shapes exactly.
    """
    try:
        import nibabel as nib  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "nibabel is required for NIfTI saving but is not available"
        ) from e

    assert Sles.shape == Sorg.shape, "Sles/Sorg shapes must match"
    _ensure_dir(out_dir)
    B = Sles.shape[0]
    assert len(ref_nii_paths) == B, "ref_nii_paths length must match batch size"
    if names is None:
        names = [f"{i:04d}" for i in range(B)]
    assert len(names) == B, "names length must match batch size"

    out_paths: List[Tuple[str, str]] = []
    Sles_np = Sles.detach().cpu().numpy()
    Sorg_np = Sorg.detach().cpu().numpy()

    # Determine spatial dims from reference volumes
    for i in range(B):
        ref_img = nib.load(ref_nii_paths[i])
        ref_data = ref_img.get_fdata()
        spatial = tuple(Sles_np[i].shape)
        # Accept either exact spatial match (3D) or 4D feature refs [X,Y,Z,D]
        ok = False
        if ref_data.shape == spatial:
            ok = True
        elif ref_data.ndim == (len(spatial) + 1) and tuple(ref_data.shape[: len(spatial)]) == spatial:
            ok = True
        if not ok:
            raise ValueError(
                f"Shape mismatch for case {i}: heatmap {spatial} vs ref {ref_data.shape}"
            )
        # Create fresh headers to avoid datatype/scaling quirks from feature refs
        les_arr = Sles_np[i].astype(np.float32, copy=False)
        org_arr = Sorg_np[i].astype(np.float32, copy=False)
        # Build fresh headers with explicit dtype to avoid nibabel dtype/scaling quirks
        les_hdr = nib.Nifti1Header()
        org_hdr = nib.Nifti1Header()
        try:
            les_hdr.set_data_dtype(np.float32)
            org_hdr.set_data_dtype(np.float32)
        except Exception:
            pass
        # Set spatial zooms from reference (first len(spatial) dims)
        try:
            zooms = ref_img.header.get_zooms()
            if zooms and len(zooms) >= len(spatial):
                les_hdr.set_zooms(tuple(zooms[: len(spatial)]))
                org_hdr.set_zooms(tuple(zooms[: len(spatial)]))
        except Exception:
            pass
        les_img = nib.Nifti1Image(les_arr, ref_img.affine, header=les_hdr)
        org_img = nib.Nifti1Image(org_arr, ref_img.affine, header=org_hdr)
        les_path = os.path.join(out_dir, f"{prefix_les}_{names[i]}.nii.gz")
        org_path = os.path.join(out_dir, f"{prefix_org}_{names[i]}.nii.gz")
        nib.save(les_img, les_path)
        nib.save(org_img, org_path)
        out_paths.append((les_path, org_path))
    return out_paths


# Optional: quick parameter search utility
@torch.no_grad()
def grid_search_params(
    feats: torch.Tensor,
    T_les: torch.Tensor,
    T_org: torch.Tensor,
    tau_grid = (0.03, 0.05, 0.07, 0.1),
    beta_grid = (5.0, 10.0, 15.0),
    agg_les: str = "lse",
    agg_org: str = "mean",
    metric_key: str = "corr",
) -> Dict[str, float]:
    """
    Returns best params minimizing `metric_key` median (default: correlation).
    """
    best = {"score": float("inf"), "tau": None, "beta": None}
    dummy = type("Dummy", (), {})()
    for tau in tau_grid:
        for beta in beta_grid:
            Sles, Sorg = PromptBankEvaluator.compute_heatmaps(
                dummy,
                feats,
                T_les=T_les,
                T_org=T_org,
                tau_les=tau,
                tau_org=tau,
                agg_les=agg_les,
                agg_org=agg_org,
                beta_les=beta,
                beta_org=beta,
                normalize=True,
            )
            ms = PromptBankEvaluator.separability_metrics(dummy, Sles, Sorg)
            import statistics as st

            vals = [m[metric_key] for m in ms if not (m[metric_key] != m[metric_key])]
            if not vals:
                continue
            med = st.median(vals)
            if med < best["score"]:
                best.update({"score": med, "tau": tau, "beta": beta})
    return best
