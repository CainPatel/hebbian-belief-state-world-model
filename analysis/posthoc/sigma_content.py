"""Study 2 post-hoc exploratory analysis: WHAT IS IN sigma?

EXPLORATORY, POST-HOC, NOT PREREGISTERED. Nothing in this script decides anything. No number here
is a criterion, and none of it can support, refute or revise H1 to H8: those decisions stand exactly
as RESULTS.md records them. This is descriptive characterization only.

The question. Study 1's H4 and Study 2 both asked "is OBJECT POSITION linearly readable from
sigma?", and the answer was no both times, with the kill criteria firing. Neither study asked the
prior question of what sigma actually holds. That question is tractable because Study 2's
exploratory structure block already measured that each head's [N, D] state is effectively LOW RANK:
8 to 10 singular components reach 90% of the squared Frobenius mass and 29 to 32 reach 99%, with a
spectral participation ratio of 2.2 to 3.0. Roughly ten meaningful directions per head is few enough
to characterize exhaustively rather than guess at.

Why the directions are interpretable. sigma accumulates rank-one outer products,

    sigma_t = sum_{s <= t} gamma^(t - s) (R(s) u_s) outer x_s,  u_s = relu(x_s W_enc) >= 0,

so the COLUMN space (the D = 64 axis) is spanned by the residual-stream vectors x_s that were
written, which live in the same space as the token embeddings, and the ROW space (the N = 2048
neuron axis) is spanned by the rope-rotated write keys, which say WHICH NEURONS store a component.
Right singular vectors can therefore be read semantically by projecting them onto the model's token
embedding matrix, and onto `lm_head` to ask which token logits the direction produces.

Two honest caveats on that decoding, neither of which is a reason not to do it. First, `x` at a
level is LayerNormed and, above level 0, has already passed through the levels below, so it is not
literally an embedding row; alignment with `model.embed.weight` is an approximation of "which token
does this direction look like", not an identity. Second, `lm_head` is applied after a further
LayerNorm, so a raw projection is not a logit; cosine against the `lm_head` columns is reported
because it is scale-free and directly comparable to the embedding side. Both views are reported
because they can and do differ.

Sign convention. An SVD component's (u, v) pair is defined only up to a joint sign flip, which would
make signed token similarities meaningless if left alone. Because the write keys u_s are ReLU
outputs (nonnegative before rope), the natural canonical choice is the one that makes the LEFT
singular vector point into the mostly-positive orthant the keys occupy: each component's sign is
fixed by requiring sum_n u[n] >= 0, and the right singular vector carries the same flip. Signs
reported below are meaningful under that convention, not arbitrary.

Sampling and batching follow `hbwm.instrument.structure.measure_sigma_structure` exactly: the same
Study 2 pair generator draw order, the same seeded `rng.choice` of `n_sample` pairs and
`pairs.subset(np.sort(take))`, and the same `batch_eps` chunking. The chunking is not optional. The
un-chunked form holds a [n_layer, B, nh, N, D] fp32 sigma, about 12.6 GB at B = 1024, and this
project has already lost a checkpoint to an OOM kill. It is exact rather than approximate: every
quantity here is per example, and every reported number is an order-independent summary. The SVDs
are taken per chunk on the host and only their compact outputs are retained, so the full
[M, nh, N, D] state, 2.1 GB at M = 1024, is never materialized.

Read-only on `runs/` and `data/` apart from the one JSON it writes to `runs/study1/results2/`.
"""

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from hbwm.device import release_memory, select_device
from hbwm.envs import tokenizer as tk
from hbwm.envs.dataset import EpisodeData
from hbwm.instrument.recorder import SigmaRecorder

# `_summary` and `participation_ratio` are imported rather than reimplemented so the percentiles and
# the concentration measure are the SAME code that produced results2/structure.json, which is what
# lets these numbers sit alongside it. Nothing in hbwm/ is touched.
from hbwm.instrument.structure import _summary, participation_ratio
from hbwm.probes.eligibility import sample_pairs
from hbwm.probes.run import Study2Config, stratified_subsample
from hbwm.train import load_checkpoint

# ROOT must contain runs/ and data/. Study 1's artifacts live in the sibling worktree, which is the
# default below; override with HBWM_ROOT. OUT_DIR receives the JSON and defaults beside Study 2's
# own aggregated results, where structure.json already lives.
_REPO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("HBWM_ROOT", _REPO / ".claude/worktrees/study1-impl"))
OUT_DIR = Path(os.environ.get("HBWM_POSTHOC_OUT", ROOT / "runs/study1/results2"))
OUT_NAME = "posthoc_sigma_content.json"

BDH_RUN = "bdh_g100_lr0.003"
DATA_DIR = ROOT / "data/grid9"
# The four BDH checkpoint-levels Study 2 measured structure on (results2/structure.json keys),
# grouped by checkpoint: seed0 is measured at both levels in ONE set of recorder passes, which is
# exact, because the pair sample does not depend on the level and only which entry of the recorder
# payload is read differs.
CKPT_LEVELS = {0: [3, 4], 1: [3], 2: [4]}
TOP_K = 10  # components kept per head; Study 2 found 8 to 10 reach 90% of the squared Frobenius mass
TOP_TOKENS = 5
TINY = 1e-30

# Observation-stream base rates on the 9x9 grid, for the reader to compare the decoded tokens
# against. These are properties of the DATA, quoted here as context only; they are not recomputed.
WINDOW_CELL_RATES = {"EMPTY": 0.843, "WALL": 0.125, "OBJ": 0.0325}


def token_name(tok: int) -> str:
    """Human name for a vocabulary id.

    Thin pass-through to `hbwm.envs.tokenizer.token_name`, which already implements the scheme
    (BOS 0, PAD 1, A_BASE 2..5, X_BASE 6..16, Y_BASE 17..27, CELL_BASE 28 so EMPTY 28, WALL 29,
    OBJ k = 30 + k). Reusing it rather than restating it keeps one definition of the names.
    """
    return tk.token_name(int(tok))


def token_family(tok: int) -> str:
    """Coarse class of a vocabulary id, for the summary histogram: BOS/PAD, ACTION, X, Y, cell type."""
    name = token_name(tok)
    if name in ("BOS", "PAD"):
        return name
    if name.startswith("A_"):
        return "ACTION"
    if name.startswith("X_"):
        return "X"
    if name.startswith("Y_"):
        return "Y"
    if name.startswith("OBJ_"):
        return "OBJ"
    return name  # EMPTY, WALL


def token_masks(data, vocab_size: int):
    """Which vocabulary entries carry trained parameters on each side of the decoding.

    Untrained columns are not a nuisance to be tidied away, they actively corrupt a cosine ranking,
    so both sides are restricted to entries the data exercises. The rule is the one
    `atlas_selectivity` already uses: keep what actually occurs, and take the count from the data
    rather than hardcoding it.

    Embedding side: tokens that occur anywhere in the split. On the 9x9 grid that is 29 of the 34
    entries, because PAD is unused and the vocabulary reserves coordinate slots up to MAX_G = 11, so
    x = 9, 10 and y = 9, 10 never appear. Their embedding rows keep their initialization, at a norm
    around 0.08 against 0.55 to 0.60 for real tokens, so a cosine against them is a normalized
    random direction.

    Output-head side: tokens that occur at an UNMASKED target position, which additionally drops BOS
    and the four action tokens, since `tokenizer.loss_mask` excludes position 0 and every action
    position and those tokens are therefore never predicted. Their `lm_head` columns are degenerate:
    see `lm_head_degeneracy`.
    """
    toks = np.asarray(data.tokens, dtype=np.int64)
    occurs = np.bincount(toks.reshape(-1), minlength=vocab_size) > 0
    tgt = toks[:, tk.loss_mask(data.L)]
    predicted = np.bincount(tgt.reshape(-1), minlength=vocab_size) > 0
    return torch.from_numpy(occurs), torch.from_numpy(predicted)


def lm_head_degeneracy(head, head_mask) -> dict:
    """Measured description of the untrained `lm_head` columns, so the restriction is auditable.

    Columns for tokens that are never a target receive no gradient toward any particular token and
    collapse onto a single shared large-norm direction: they exist only to be pushed down. Reporting
    their pairwise cosine and their norm relative to the trained columns shows why an unrestricted
    ranking through `lm_head` is meaningless rather than merely noisy.
    """
    h = head.to(torch.float64)
    hn = torch.nn.functional.normalize(h, dim=0)
    keep = head_mask.clone()
    if int((~keep).sum()) < 2:
        return {"n_untrained_columns": int((~keep).sum()), "mean_abs_cos_among_untrained": None}
    g = (hn[:, ~keep].T @ hn[:, ~keep]).abs()
    n = g.shape[0]
    norms = h.norm(dim=0)
    return {
        "n_untrained_columns": int(n),
        "untrained_tokens": [token_name(i) for i in torch.nonzero(~keep).flatten().tolist()],
        "mean_abs_cos_among_untrained": float((g.sum() - g.diagonal().sum()) / (n * (n - 1))),
        "median_norm_untrained": float(norms[~keep].median()),
        "median_norm_trained": float(norms[keep].median()),
    }


def study2_pairs(cfg: Study2Config):
    """The `p_tr` that `run_probes_study2` hands to `measure_sigma_structure`.

    The generator's draw order is load-bearing: `sample_pairs` is called on probe_train, probe_val
    and probe_test in that order off the SAME rng before `stratified_subsample` draws from it, so
    reproducing p_tr means reproducing all four calls, not just the first.
    """
    rng = np.random.default_rng(cfg.seed)
    splits = ("probe_train", "probe_val", "probe_test")
    d_tr, d_va, d_te = (EpisodeData(str(DATA_DIR), s) for s in splits)
    p_tr_all, _p_va, _p_te = (sample_pairs(d, rng, cfg.per_obj) for d in (d_tr, d_va, d_te))
    return d_tr, stratified_subsample(p_tr_all, cfg.n_train, rng)


def structure_subsample(pairs, n_sample: int, seed: int):
    """Exactly `measure_sigma_structure`'s own subsample: fresh rng(seed), choice, sorted subset."""
    rng = np.random.default_rng(seed)
    take = rng.choice(len(pairs), size=min(n_sample, len(pairs)), replace=False)
    return pairs.subset(np.sort(take))


def svd_block(blk, top_k: int):
    """SVD of one host-side block of states. blk: [b, nh, N, D] fp32.

    Returns (singular values [b, nh, D], right vectors [b, nh, D, k], left participation ratios
    [b, nh, k]). Float32 to match `effective_rank`, which produced structure.json's spectrum; the
    derived shares are taken in float64 there and here. The joint sign of each (u, v) pair is fixed
    by sum_n u[n] >= 0 (see the module docstring), applied to v so its token signs are meaningful.
    The left vectors are summarized to a participation ratio and dropped: keeping U for the whole
    sample would cost as much as keeping sigma itself.
    """
    u, s, vh = torch.linalg.svd(blk, full_matrices=False)  # [b,nh,N,D], [b,nh,D], [b,nh,D,D]
    uk = u[..., :top_k]  # [b, nh, N, k]
    sign = torch.where(uk.sum(dim=-2) >= 0, 1.0, -1.0)  # [b, nh, k]
    v = vh[..., :top_k, :].transpose(-1, -2) * sign.unsqueeze(-2)  # [b, nh, D, k], columns are v_c
    # participation ratio over the 2048 neurons of the mass u^2, the same functional form
    # `row_norm_stats` applies to squared row norms: 1 means one neuron, N means perfectly spread.
    u_pr = participation_ratio(uk.to(torch.float64).pow(2).transpose(-1, -2))  # [b, nh, k]
    return s.clone(), v.contiguous(), u_pr.to(torch.float32)


@torch.no_grad()
def collect(model, data, p, levels, device, batch_eps: int, top_k: int, svd_batch: int):
    """One set of recorder passes; returns per level (singular values, right vectors, left PRs).

    Mirrors `measure_sigma_structure`'s pass, minus its write-mass accumulator: nothing here needs a
    per-step quantity, so the recorder is given the sampled positions explicitly and the callback
    fires only there. Chunked at `batch_eps` episodes, and each chunk's states are reduced to their
    SVDs and released before the next chunk allocates.
    """
    obs_pos = tk.obs_positions(data.L)
    acc = {lvl: {"sv": [], "v": [], "u_pr": []} for lvl in levels}
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
        buf = {lvl: [] for lvl in levels}

        def fn(pos, payload, by_pos=by_pos, buf=buf):
            r = torch.as_tensor(by_pos[pos], device=device)
            for lvl in levels:
                # payload["sigma"] aliases live state that the next step() mutates: index and copy.
                buf[lvl].append(payload["sigma"][lvl][r].float().cpu())

        rec.run(tokens, sorted(by_pos), fn)
        del tokens, fn
        release_memory(device)
        for lvl in levels:
            states = torch.cat(buf[lvl]) if buf[lvl] else torch.empty(0)
            buf[lvl].clear()
            for j0 in range(0, len(states), svd_batch):
                sv, v, u_pr = svd_block(states[j0 : j0 + svd_batch], top_k)
                acc[lvl]["sv"].append(sv)
                acc[lvl]["v"].append(v)
                acc[lvl]["u_pr"].append(u_pr)
            del states
        del buf
        release_memory(device)
    out = {}
    for lvl in levels:
        a = acc[lvl]
        out[lvl] = (torch.cat(a["sv"]), torch.cat(a["v"]), torch.cat(a["u_pr"]))
        a["sv"].clear(), a["v"].clear(), a["u_pr"].clear()
    return out


def spectrum_stats(sv, top_k: int) -> dict:
    """Measurement 1: singular value shares and top-component concentration. sv: [M, nh, D]."""
    e = sv.to(torch.float64).pow(2)
    tot = e.sum(dim=-1, keepdim=True).clamp(min=TINY)
    share = e / tot
    cum = torch.cumsum(share, dim=-1)
    s = sv.to(torch.float64)
    ratio_12 = s[..., 0] / s[..., 1].clamp(min=TINY)
    return {
        "top1_share": _summary(share[..., 0]),
        "top1_over_top2_singular_value": _summary(ratio_12),
        f"share_top_{top_k}": _summary(share[..., :top_k].sum(dim=-1)),
        "n_components_90pct": _summary((cum < 0.90).sum(dim=-1) + 1),
        "n_components_99pct": _summary((cum < 0.99).sum(dim=-1) + 1),
        "spectral_participation_ratio": _summary(participation_ratio(e)),
        "component_share_median": [float(share[..., c].median()) for c in range(top_k)],
        "component_share_by_head": {
            f"h{h}": [float(share[:, h, c].median()) for c in range(top_k)]
            for h in range(share.shape[1])
        },
        "n_singular_values": int(sv.shape[-1]),
    }


def mean_abs_offdiag_cos(v) -> float:
    """Mean |cos| over all distinct ordered pairs of rows. v: [M, D], rows already unit norm."""
    m = v.shape[0]
    if m < 2:
        return float("nan")
    g = (v @ v.T).abs()
    return float((g.sum() - g.diagonal().sum()) / (m * (m - 1)))


def stability_stats(vecs, seed: int) -> dict:
    """Measurement 3: is the top direction shared across examples, or example-specific?

    vecs: [M, nh, D, k] right singular vectors. Reports the mean pairwise |cos| between DIFFERENT
    examples' top-1 right singular vectors, per head, against a random-direction baseline drawn in
    the same dimension with the same number of vectors, so the two are directly comparable. |cos| is
    used because a shared axis is a shared axis regardless of which end of it an example lands on.
    A value at the baseline means every example has its own basis; well above it means sigma stores
    an example-independent set of directions.
    """
    v = torch.nn.functional.normalize(vecs.to(torch.float64), dim=-2)
    m, nh, _, k = v.shape
    rng = np.random.default_rng(seed)
    rand = torch.from_numpy(rng.standard_normal((m, v.shape[2])))
    rand = torch.nn.functional.normalize(rand, dim=-1)
    base = mean_abs_offdiag_cos(rand)
    per_head = {f"h{h}": mean_abs_offdiag_cos(v[:, h, :, 0]) for h in range(nh)}
    per_comp = {
        f"h{h}": [mean_abs_offdiag_cos(v[:, h, :, c]) for c in range(k)] for h in range(nh)
    }
    vals = [x for x in per_head.values() if np.isfinite(x)]
    return {
        "top1_mean_abs_cos_by_head": per_head,
        "top1_mean_abs_cos_mean_over_heads": float(np.mean(vals)) if vals else float("nan"),
        "random_baseline_mean_abs_cos": base,
        "top1_ratio_to_baseline_by_head": {
            h: (x / base if base > 0 else float("nan")) for h, x in per_head.items()
        },
        "mean_abs_cos_by_head_by_component": per_comp,
        "n_examples": int(m),
        "n_dims": int(v.shape[2]),
        "note": ("|cos| between different examples' top-1 right singular vectors; the baseline is "
                 "the same statistic over the same number of random unit vectors in the same "
                 "dimension, so at the baseline the basis is example-specific."),
    }


def top_tokens(scores, k: int, mask=None) -> list:
    """The k entries of `scores` ([V] float64) with the largest magnitude, with their signs.

    `mask` restricts the ranking to the vocabulary entries it selects; the reported ids and names
    are always the true vocabulary ids, never positions within the restricted set.
    """
    ids = torch.arange(scores.shape[0]) if mask is None else torch.nonzero(mask).flatten()
    s = scores[ids]
    idx = ids[torch.argsort(s.abs(), descending=True)[:k]]
    return [{"token": int(i), "name": token_name(int(i)), "value": float(scores[i])} for i in idx]


def decode_directions(vecs, emb, head, top_k: int, n_tokens: int, emb_mask, head_mask):
    """Measurement 2: read each component's consensus right singular vector semantically.

    vecs: [M, nh, D, k]. emb: [V, D] token embeddings. head: [D, V] output head.

    A per-example direction is not a thing to decode on its own, so for each (head, component) the
    CONSENSUS direction is taken as the leading eigenvector of the second moment matrix
    (1/M) sum_e v_e v_e^T, which is the axis the examples' directions concentrate on, and its
    eigenvalue share says how well defined that consensus is. The examples' signs are already
    canonicalized in `svd_block`, so the consensus sign is fixed from the data rather than by
    convention. The modal per-example top-1 token is reported alongside as a check that the
    consensus is not an artifact of averaging.

    `embed_top_tokens` and `lm_head_top_tokens` are ranked over the trained vocabulary entries only
    (see `token_masks`); the unrestricted rankings are kept beside them under `_all_vocab` so the
    effect of the restriction is visible rather than hidden. Returns (entries, [min cos, max cos])
    where the range is over the FULL cosine vectors, as a bound check.
    """
    v = torch.nn.functional.normalize(vecs.to(torch.float64), dim=-2)
    m, nh, _, k = v.shape
    emb_n = torch.nn.functional.normalize(emb.to(torch.float64), dim=-1)  # [V, D]
    head_n = torch.nn.functional.normalize(head.to(torch.float64), dim=0)  # [D, V]
    emb_rank = emb_n[emb_mask]  # rows used for the per-example modal token
    emb_ids = torch.nonzero(emb_mask).flatten()
    out, lo, hi = [], float("inf"), float("-inf")
    for h in range(nh):
        for c in range(min(top_k, k)):
            vc = v[:, h, :, c]  # [M, D]
            second = (vc.T @ vc) / max(m, 1)
            evals, evecs = torch.linalg.eigh(second)
            ref = evecs[:, -1]
            if float((vc @ ref).sum()) < 0:  # align the consensus with the canonicalized examples
                ref = -ref
            cos_emb = emb_n @ ref  # [V]
            cos_head = ref @ head_n  # [V]
            lo = min(lo, float(cos_emb.min()), float(cos_head.min()))
            hi = max(hi, float(cos_emb.max()), float(cos_head.max()))
            per_ex = emb_ids[(emb_rank @ vc.T).abs().argmax(dim=0)]  # [M] each example's top-1
            counts = Counter(int(t) for t in per_ex.tolist())
            modal = [
                {"token": t, "name": token_name(t), "frac": n / max(m, 1)}
                for t, n in counts.most_common(3)
            ]
            out.append({
                "head": h, "component": c,
                "consensus_eigenvalue_share": float(evals[-1] / evals.sum().clamp(min=TINY)),
                "mean_abs_cos_to_consensus": float((vc @ ref).abs().mean()),
                "embed_top_tokens": top_tokens(cos_emb, n_tokens, emb_mask),
                "lm_head_top_tokens": top_tokens(cos_head, n_tokens, head_mask),
                "lm_head_top_raw_projection": top_tokens(
                    ref @ head.to(torch.float64), n_tokens, head_mask),
                "embed_top_tokens_all_vocab": top_tokens(cos_emb, n_tokens),
                "lm_head_top_tokens_all_vocab": top_tokens(cos_head, n_tokens),
                "per_example_modal_top_token": modal,
            })
    return out, [lo, hi]


def family_histogram(decoded: list, share_by_head: dict) -> dict:
    """Energy-weighted histogram of the consensus top-1 token family, over heads and components.

    Each (head, component) contributes its median squared-singular-value share, so a direction that
    carries almost no mass cannot dominate the picture. Descriptive only.
    """
    tot, fam, tok = 0.0, Counter(), Counter()
    for d in decoded:
        w = share_by_head[f"h{d['head']}"][d["component"]]
        name = d["embed_top_tokens"][0]["name"]
        fam[token_family(d["embed_top_tokens"][0]["token"])] += w
        tok[name] += w
        tot += w
    if tot <= 0:
        return {"by_family": {}, "by_token": {}, "total_weight": 0.0}
    return {
        "by_family": {k: v / tot for k, v in fam.most_common()},
        "by_token": {k: v / tot for k, v in tok.most_common(8)},
        "total_weight": tot,
        "weighting": "median squared singular value share of the component",
    }


def rowspace_stats(u_pr, n_rows: int) -> dict:
    """Measurement 4: how concentrated is the top component's left singular vector over neurons?"""
    pr = u_pr.to(torch.float64)
    return {
        "top1_participation_ratio": _summary(pr[..., 0]),
        "top1_participation_fraction": _summary(pr[..., 0] / n_rows),
        "participation_ratio_median_by_component": [
            float(pr[..., c].median()) for c in range(pr.shape[-1])
        ],
        "top1_participation_ratio_by_head": {
            f"h{h}": _summary(pr[:, h, 0]) for h in range(pr.shape[1])
        },
        "n_rows": int(n_rows),
        "note": ("participation ratio of u^2 over the 2048 neurons: 1 means the component lives on "
                 "one neuron, n_rows means it is spread evenly."),
    }


def interpretation(hist: dict, stab: dict, spec: dict, rows: dict) -> list:
    """Plain statements of what the decoding shows. Reads the numbers; forces no story either way."""
    lines = []
    fam = hist.get("by_family", {})
    ranked = ", ".join(f"{k} {v:.1%}" for k, v in list(fam.items())[:5]) or "none"
    lines.append(f"Consensus top-1 embedding token by family, weighted by component energy: {ranked}.")
    obj = fam.get("OBJ", 0.0)
    cells = fam.get("EMPTY", 0.0) + fam.get("WALL", 0.0)
    coords = fam.get("X", 0.0) + fam.get("Y", 0.0)
    if obj >= max(cells, coords):
        lines.append(
            f"Object tokens carry the largest share ({obj:.1%}) of the top directions, above cell "
            f"tokens ({cells:.1%}) and coordinate tokens ({coords:.1%}). The leading directions of "
            "sigma align with object identity tokens.")
    else:
        lines.append(
            f"Object tokens carry {obj:.1%} of the top directions, against {cells:.1%} for "
            f"EMPTY and WALL and {coords:.1%} for X and Y coordinates. The leading directions of "
            "sigma align with cell-type and coordinate tokens rather than object tokens.")
        lines.append(
            "That would be consistent with sigma storing mostly the high-frequency structure of "
            f"the observation stream ({WINDOW_CELL_RATES['EMPTY']:.1%} of window cells are empty, "
            f"{WINDOW_CELL_RATES['WALL']:.1%} wall, {WINDOW_CELL_RATES['OBJ']:.2%} objects) rather "
            "than object identity or location. It is not evidence for that reading on its own: a "
            "shared direction aligned with EMPTY says what dominates the leading components, not "
            "that nothing else is present in the remaining ones.")
    base = stab["random_baseline_mean_abs_cos"]
    got = stab["top1_mean_abs_cos_mean_over_heads"]
    if np.isfinite(got) and np.isfinite(base) and base > 0:
        if got > 3 * base:
            lines.append(
                f"Top-1 directions are shared across examples: mean pairwise |cos| {got:.3f} "
                f"against a random baseline of {base:.3f} ({got / base:.1f}x). sigma holds an "
                "example-independent set of directions, not a per-example basis.")
        elif got > 1.5 * base:
            lines.append(
                f"Top-1 directions are partly shared: mean pairwise |cos| {got:.3f} against a "
                f"random baseline of {base:.3f} ({got / base:.1f}x).")
        else:
            lines.append(
                f"Top-1 directions are close to example-specific: mean pairwise |cos| {got:.3f} "
                f"against a random baseline of {base:.3f} ({got / base:.1f}x).")
    lines.append(
        f"Median top-1 singular value share {spec['top1_share']['median']:.3f}, with the top "
        f"component's left singular vector spread over a participation ratio of "
        f"{rows['top1_participation_ratio']['median']:.1f} of {rows['n_rows']} neurons.")
    lines.append(
        "EXPLORATORY and post-hoc. Nothing above revises H1 to H8 or any preregistered decision.")
    return lines


def analyze(sv, vecs, u_pr, model, masks, top_k: int, n_tokens: int, seed: int) -> dict:
    emb_mask, head_mask = masks
    spec = spectrum_stats(sv, top_k)
    stab = stability_stats(vecs, seed)
    rows = rowspace_stats(u_pr, model.hcfg.n_neurons)
    emb = model.embed.weight.detach().cpu()
    head = model.lm_head.detach().cpu()
    decoded, cos_range = decode_directions(vecs, emb, head, top_k, n_tokens, emb_mask, head_mask)
    hist = family_histogram(decoded, spec["component_share_by_head"])
    return {
        "spectrum": spec,
        "semantic_decoding": decoded,
        "consensus_token_histogram": hist,
        "stability": stab,
        "row_space": rows,
        "interpretation": interpretation(hist, stab, spec, rows),
        "decoding_vocabulary": {
            "embed_tokens": [token_name(i) for i in torch.nonzero(emb_mask).flatten().tolist()],
            "lm_head_tokens": [token_name(i) for i in torch.nonzero(head_mask).flatten().tolist()],
            "n_embed_tokens": int(emb_mask.sum()), "n_lm_head_tokens": int(head_mask.sum()),
            "lm_head_degeneracy": lm_head_degeneracy(head, head_mask),
        },
        "n_examples": int(sv.shape[0]), "n_head": int(sv.shape[1]), "top_k": int(top_k),
        "cosine_range_observed": [float(cos_range[0]), float(cos_range[1])],
        "cosines_in_unit_range": bool(cos_range[0] >= -1.0 - 1e-9 and cos_range[1] <= 1.0 + 1e-9),
        "singular_values_shape": list(sv.shape),
        "all_finite": bool(torch.isfinite(sv).all() and torch.isfinite(vecs).all()),
        "window_cell_base_rates": WINDOW_CELL_RATES,
        "exploratory": True,
    }


def print_report(key: str, r: dict, n_show: int) -> None:
    spec, stab, rows = r["spectrum"], r["stability"], r["row_space"]
    share_k = spec["share_top_%d" % r["top_k"]]["median"]
    print(f"\n=== {key}  (M={r['n_examples']} examples, {r['n_head']} heads, k={r['top_k']}) ===")
    print(f"  spectrum: top1 share {spec['top1_share']['median']:.4f} "
          f"[{spec['top1_share']['p10']:.4f},{spec['top1_share']['p90']:.4f}]  "
          f"s1/s2 {spec['top1_over_top2_singular_value']['median']:.2f}  "
          f"top-{r['top_k']} share {share_k:.4f}  "
          f"k90 {spec['n_components_90pct']['median']:.0f}  "
          f"k99 {spec['n_components_99pct']['median']:.0f}  "
          f"spectral PR {spec['spectral_participation_ratio']['median']:.2f}")
    print(f"  row space: top-1 left vector participation ratio "
          f"{rows['top1_participation_ratio']['median']:.1f} of {rows['n_rows']} neurons "
          f"({rows['top1_participation_fraction']['median']:.4f} of them)")
    print(f"  stability: top-1 mean pairwise |cos| across examples "
          f"{stab['top1_mean_abs_cos_mean_over_heads']:.4f} vs random baseline "
          f"{stab['random_baseline_mean_abs_cos']:.4f}  per head "
          + " ".join(f"{h}={x:.3f}" for h, x in stab["top1_mean_abs_cos_by_head"].items()))
    print(f"  {'head/comp':10s} {'share':>7s} {'consens':>7s} "
          f"{'embed top tokens':40s} {'lm_head top tokens':40s}")
    for d in r["semantic_decoding"]:
        if d["component"] >= n_show:
            continue
        share = spec["component_share_by_head"][f"h{d['head']}"][d["component"]]
        fmt = lambda ts: " ".join(  # noqa: E731
            f"{t['name']}{'+' if t['value'] >= 0 else '-'}{abs(t['value']):.2f}" for t in ts)
        print(f"  h{d['head']}/c{d['component']:<7d} {share:7.4f} "
              f"{d['consensus_eigenvalue_share']:7.3f} "
              f"{fmt(d['embed_top_tokens'][:3]):40s} {fmt(d['lm_head_top_tokens'][:3]):40s}")
    hist = r["consensus_token_histogram"]
    print("  consensus top-1 token, energy weighted, by family: "
          + " ".join(f"{k}={v:.1%}" for k, v in hist["by_family"].items()))
    print("  by token: " + " ".join(f"{k}={v:.1%}" for k, v in hist["by_token"].items()))
    dv = r["decoding_vocabulary"]
    deg = dv["lm_head_degeneracy"]
    print(f"  decoding vocabulary: {dv['n_embed_tokens']} embed tokens, "
          f"{dv['n_lm_head_tokens']} lm_head tokens; "
          f"{deg['n_untrained_columns']} untrained lm_head columns "
          f"({', '.join(deg['untrained_tokens'])}) collapsed onto one direction, "
          f"mean |cos| {deg['mean_abs_cos_among_untrained']:.4f}, "
          f"norm {deg['median_norm_untrained']:.2f} vs {deg['median_norm_trained']:.2f} trained")
    print(f"  checks: singular values {r['singular_values_shape']}  finite={r['all_finite']}  "
          f"cosines in [-1,1]={r['cosines_in_unit_range']} "
          f"(observed [{r['cosine_range_observed'][0]:.4f},{r['cosine_range_observed'][1]:.4f}])")
    for line in r["interpretation"]:
        print(f"  * {line}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, nargs="*", default=sorted(CKPT_LEVELS))
    ap.add_argument("--levels", type=int, nargs="*", default=None, help="override the per-seed levels")
    ap.add_argument("--n-sample", type=int, default=Study2Config.structure_n_sample)
    ap.add_argument("--batch-eps", type=int, default=Study2Config.batch_eps)
    ap.add_argument("--svd-batch", type=int, default=16, help="examples per host-side SVD call")
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--top-tokens", type=int, default=TOP_TOKENS)
    ap.add_argument("--show-components", type=int, default=3, help="components printed per head")
    ap.add_argument("--seed", type=int, default=Study2Config.seed)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=str(OUT_DIR / OUT_NAME))
    ap.add_argument("--no-write", action="store_true", help="print only; for the validation slice")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace the output JSON instead of merging this run's keys into it")
    args = ap.parse_args()

    device = select_device(args.device)
    cfg = Study2Config(seed=args.seed, batch_eps=args.batch_eps)
    t0 = time.time()
    data, p_tr = study2_pairs(cfg)
    p = structure_subsample(p_tr, args.n_sample, cfg.seed)
    masks = token_masks(data, tk.VOCAB_SIZE)
    print(f"device={device} probe_train pairs={len(p_tr)} sampled={len(p)} "
          f"episodes={len(np.unique(p.ep))} batch_eps={args.batch_eps} top_k={args.top_k} "
          f"T={data.T} ({time.time() - t0:.1f}s to build pairs)")

    out, rows = {}, []
    for seed in args.seeds:
        levels = args.levels if args.levels is not None else CKPT_LEVELS[seed]
        ckpt = ROOT / "runs/study1" / BDH_RUN / f"seed{seed}/ckpt.pt"
        model, _, meta = load_checkpoint(ckpt, device)
        t1 = time.time()
        got = collect(model, data, p, levels, device, args.batch_eps, args.top_k, args.svd_batch)
        secs = time.time() - t1
        for lvl in levels:
            sv, vecs, u_pr = got[lvl]
            key = f"bdh_g100/seed{seed}/sigma_content_L{lvl}"
            r = analyze(sv, vecs, u_pr, model, masks, args.top_k, args.top_tokens, cfg.seed)
            r.update({"level": lvl, "seed": seed, "run": BDH_RUN, "ckpt": str(ckpt),
                      "ckpt_step": int(meta["step"]), "decay_gamma": float(model.hcfg.decay_gamma),
                      "n_sample": args.n_sample, "sample_seed": cfg.seed,
                      "batch_eps": args.batch_eps, "elapsed_s": round(secs, 1)})
            out[key] = r
            rows.append((key, r))
        del got, model
        release_memory(device)
        print(f"  seed{seed} levels={levels}: {secs:.1f}s")

    for key, r in rows:
        print_report(key, r, args.show_components)

    if not args.no_write:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Merge by default: one checkpoint-level is a full set of recorder passes, so a run that
        # covers a subset of seeds must not discard the keys an earlier run already paid for. Keys
        # are per checkpoint-level and this run's own keys win.
        merged = {}
        if path.exists() and not args.overwrite:
            merged = json.loads(path.read_text())
        merged.update(out)
        path.write_text(json.dumps(merged, indent=2) + "\n")
        print(f"\nwrote {path} ({len(merged)} checkpoint-levels)")
    print(f"total wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
