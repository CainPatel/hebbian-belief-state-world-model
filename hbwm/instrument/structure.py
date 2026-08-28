"""Exploratory descriptive measurements of sigma (spec section 4.8).

NOT PREREGISTERED. Nothing in this module decides anything: no threshold here is a criterion, and no
result here can support or refute H5 to H8. Study 1's H4 asked whether the *decodable* signal
concentrates in few features under a flat linear probe (median k90 = 524,288, RESULTS.md); these
measurements ask whether sigma is *structurally* sparse or low rank, which H4 never tested.
"""

import numpy as np
import torch

from hbwm.device import release_memory

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


@torch.no_grad()
def measure_sigma_structure(model, data, pairs, level: int, n_sample: int = 1024, seed: int = 0,
                            device=None, atlas_episodes: int = 500, batch_eps: int = 32) -> dict:
    """All five spec 4.8 measurements for one checkpoint-level. EXPLORATORY: decides nothing.

    Measurements 1, 2 and 4 are per example and use a fixed seeded subsample of `n_sample` eligible
    pairs, because measurement 2 needs one SVD of an [N, D] matrix per head per example.

    The sampled episodes are driven `batch_eps` at a time (32, as in `PRESETS["study1"]`). One pass
    over all of them at the preregistered n_sample = 1024 would hold a [n_layer, B, nh, N, D] fp32
    sigma -- about 12.6 GB at B = 1024 -- for the whole pass, on top of the probe modules still
    resident from the same level. Chunking is numerically exact rather than an approximation: the
    write-mass accumulator is per-episode independent, every measurement is per example, and every
    reported number is an order-independent summary. Each chunk therefore starts from its OWN
    zero-initialized accumulator; `w` never spans chunks, whose episodes are unrelated.
    """
    from hbwm.bdh.upstream.bdh import Attention
    from hbwm.envs import tokenizer as tk
    from hbwm.instrument.atlas import build_atlas
    from hbwm.instrument.recorder import SigmaRecorder

    device = device if device is not None else next(model.parameters()).device
    rng = np.random.default_rng(seed)
    take = rng.choice(len(pairs), size=min(n_sample, len(pairs)), replace=False)
    p = pairs.subset(np.sort(take))
    gamma2 = float(model.hcfg.decay_gamma) ** 2
    obs_pos = tk.obs_positions(data.L)
    sig, xs, wmass = [], [], []
    all_eps = np.unique(p.ep)
    rec = SigmaRecorder(model)
    for b0 in range(0, len(all_eps), batch_eps):
        eps = all_eps[b0 : b0 + batch_eps]
        row_of_ep = {int(e): i for i, e in enumerate(eps)}
        by_pos = {}
        for i in range(len(p)):
            r_i = row_of_ep.get(int(p.ep[i]))
            if r_i is not None:
                by_pos.setdefault(int(obs_pos[p.t[i]]), []).append(r_i)
        tokens = torch.from_numpy(data.tokens[eps].astype(np.int64)).to(device)
        w = torch.zeros(model.hcfg.n_head, model.hcfg.n_neurons, len(eps), device=device)

        def fn(pos, payload, by_pos=by_pos, w=w):
            u = payload["x_sparse"][level]  # B, nh, N
            q = Attention.rope((float(pos) * model.attn.freqs).view(1, 1, -1), u)
            xn = payload["resid"][level].pow(2).sum(-1)  # B
            w.mul_(gamma2).add_(torch.einsum("bhn,b->hnb", q.pow(2), xn))
            if pos in by_pos:
                r = torch.as_tensor(by_pos[pos], device=device)
                sig.append(payload["sigma"][level][r].float().cpu())
                xs.append(u[r].float().cpu())
                wmass.append(w.permute(2, 0, 1)[r].float().cpu())

        # positions=None on purpose: the write-mass accumulator must run at EVERY step, not only at
        # the sampled ones, so the callback fires each step and stores only where `pos in by_pos`.
        rec.run(tokens, None, fn)
        del tokens, w, fn  # the chunk's device state, before the next chunk allocates its own
        release_memory(device)
    sigma = torch.cat(sig)
    _, tok_mean = build_atlas(model, data, n_episodes=atlas_episodes, device=device,
                              return_means=True)
    counts = np.bincount(data.tokens[:atlas_episodes].reshape(-1).astype(np.int64),
                         minlength=model.hcfg.vocab_size)
    return {"row_norm": row_norm_stats(sigma), "effective_rank": effective_rank(sigma),
            "activation": activation_sparsity(torch.cat(xs)),
            "write_concentration": write_concentration(torch.cat(wmass)),
            "atlas_selectivity": atlas_selectivity(tok_mean[level], counts),
            "n_sample": int(sigma.shape[0]), "level": level, "exploratory": True}
