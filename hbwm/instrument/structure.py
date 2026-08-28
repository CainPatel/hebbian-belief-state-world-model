"""Exploratory descriptive measurements of sigma (spec section 4.8).

NOT PREREGISTERED. Nothing in this module decides anything: no threshold here is a criterion, and no
result here can support or refute H5 to H8. Study 1's H4 asked whether the *decodable* signal
concentrates in few features under a flat linear probe (median k90 = 524,288, RESULTS.md); these
measurements ask whether sigma is *structurally* sparse or low rank, which H4 never tested.
"""

import numpy as np
import torch

TOP_FRACTIONS = (0.01, 0.10)  # spec 4.8 measurement 4


def _summary(x) -> dict:
    """Median with the 10th and 90th percentiles across the example axis (spec 4.8 sampling)."""
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    return {"median": float(np.median(a)), "p10": float(np.quantile(a, 0.10)),
            "p90": float(np.quantile(a, 0.90)), "min": float(a.min()), "max": float(a.max()),
            "mean": float(a.mean()), "n": int(a.size)}


NAN_SUMMARY = {"median": float("nan"), "p10": float("nan"), "p90": float("nan"),
               "min": float("nan"), "max": float("nan"), "mean": float("nan"), "n": 0}


def participation_ratio(a):
    """(sum a)^2 / sum a^2 over the last axis: an effective count of entries carrying mass."""
    a = torch.as_tensor(a, dtype=torch.float64)
    num = a.sum(dim=-1) ** 2
    den = (a**2).sum(dim=-1)
    return torch.where(den > 0, num / den, torch.zeros_like(den))


def row_norm_stats(sigma) -> dict:
    """Spec 4.8 measurement 1. sigma: [B, nh, N, D], unstandardized."""
    s = torch.as_tensor(sigma, dtype=torch.float32)
    norms = s.norm(dim=-1)  # [B, nh, N]
    a = norms.to(torch.float64) ** 2
    mx = norms.amax(dim=-1, keepdim=True).clamp(min=1e-30)
    return {
        "participation_ratio": _summary(participation_ratio(a)),
        "frac_below_1pct_of_max": _summary((norms < 0.01 * mx).to(torch.float64).mean(dim=-1)),
        "frac_below_10pct_of_max": _summary((norms < 0.10 * mx).to(torch.float64).mean(dim=-1)),
        "row_norm_median": _summary(norms.median(dim=-1).values),
        "row_norm_max": _summary(norms.amax(dim=-1)),
        "n_rows": int(s.shape[-2]),
    }


def effective_rank(sigma) -> dict:
    """Spec 4.8 measurement 2. At most min(N, D) = D singular values are nonzero."""
    s = torch.as_tensor(sigma, dtype=torch.float32)
    sv = torch.linalg.svdvals(s).to(torch.float64)  # [B, nh, min(N, D)]
    e = sv**2
    tot = e.sum(dim=-1, keepdim=True).clamp(min=1e-30)
    cum = torch.cumsum(e, dim=-1) / tot
    return {
        "n_components_90pct": _summary((cum < 0.90).sum(dim=-1) + 1),
        "n_components_99pct": _summary((cum < 0.99).sum(dim=-1) + 1),
        "participation_ratio": _summary(participation_ratio(e)),
        "top1_share": _summary(e[..., 0] / tot.squeeze(-1)),
        "n_singular_values": int(sv.shape[-1]),
    }


def write_concentration(w) -> dict:
    """Spec 4.8 measurement 4. w: [B, nh, N] accumulated squared write mass per row."""
    w = torch.as_tensor(w, dtype=torch.float64)
    n = w.shape[-1]
    srt = torch.sort(w, dim=-1, descending=True).values
    tot = srt.sum(dim=-1).clamp(min=1e-30)
    out = {}
    for f in TOP_FRACTIONS:
        k = max(1, int(n * f))  # floor, per spec 4.8 measurement 4 (20 and 204 rows at N = 2048)
        out[f"share_top_{int(f * 100)}pct"] = _summary(srt[..., :k].sum(dim=-1) / tot)
    out["participation_ratio"] = _summary(participation_ratio(w))
    out["n_rows"] = n
    return out


def activation_sparsity(x_sparse) -> dict:
    """Spec 4.8 measurement 3: ReLU zero fraction of the write key. Contrast only."""
    x = torch.as_tensor(x_sparse, dtype=torch.float32)
    return {"zero_fraction": _summary((x == 0).to(torch.float64).mean(dim=-1)),
            "n_neurons": int(x.shape[-1])}


def atlas_selectivity(tok_mean, token_counts) -> dict:
    """Spec 4.8 measurement 5: peakedness of each neuron's token-conditional mean activation.

    tok_mean: [V, nh, N] for ONE level. Tokens with zero `token_counts` are dropped (PAD is unused, so
    33 of the 34 vocabulary entries survive). `x_sparse` is a ReLU output, so the profile is
    nonnegative and normalizes to a distribution.
    """
    m = torch.as_tensor(tok_mean, dtype=torch.float64)
    keep = torch.as_tensor(np.asarray(token_counts) > 0)
    m = m[keep]  # [V', nh, N]
    v = int(m.shape[0])
    tot = m.sum(dim=0)  # [nh, N]
    live = tot > 0
    p = torch.where(live.unsqueeze(0), m / tot.clamp(min=1e-30), torch.zeros_like(m))
    max_share = p.amax(dim=0)[live]
    ent = -(p * torch.log(p.clamp(min=1e-30))).sum(dim=0)[live] / np.log(v)
    frac_peaked = float((max_share > 0.5).to(torch.float64).mean()) if max_share.numel() else float("nan")
    return {"max_share": _summary(max_share) if max_share.numel() else dict(NAN_SUMMARY),
            "normalized_entropy": _summary(ent) if ent.numel() else dict(NAN_SUMMARY),
            "frac_max_share_above_half": frac_peaked,
            "n_tokens": v, "n_neurons_live": int(live.sum())}
