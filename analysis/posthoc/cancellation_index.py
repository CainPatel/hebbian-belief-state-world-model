"""Study 2 post-hoc exploratory analysis: do BDH's writes into sigma CANCEL each other?

EXPLORATORY, POST-HOC, NOT PREREGISTERED. Nothing in this script decides anything. No number here
is a criterion, and none of it can support, refute or revise H4 or any of H5 to H8: those decisions
stand exactly as RESULTS.md records them. This only computes a descriptive quantity that spec
section 4.8 measurement 4 defines but the shipped instrument never reported.

The question. Study 2's exploratory structure block found sigma is structurally sparse (row-norm
participation ratio 73 to 111 of 2048 rows) and low rank (8 to 10 components carry 90% of the
squared Frobenius mass, of 64), yet position is not linearly decodable from it. One explanation is
DESTRUCTIVE SUPERPOSITION: `bdh_g100` runs at decay gamma = 1.0, so nothing is ever forgotten and
all 1,164 token steps of an episode sum into roughly a hundred effective rows at equal weight.
Later writes may then cancel earlier ones.

The diagnostic, quoting spec 4.8 measurement 4: "The ratio of measurement 1's sum of a to
measurement 4's sum of w is a cancellation index: above 1 when successive writes into a row align,
below 1 when they cancel."

  a[h, n] = ||sigma[h, n, :]||^2                      realized squared mass in row n of head h
  w[h, n] = sum_{s <= t} gamma^(2(t - s)) q_s[h, n]^2 ||x_s||^2   squared mass ROUTED into that row

Both are already computed inside `hbwm.instrument.structure.measure_sigma_structure`, whose JSON
keeps only participation ratios and top-k shares and drops the raw sums. This script recomputes
them under exactly that function's conventions -- same pair sampling, same rope-reconstructed write
key q, same accumulator ordering -- and reports the ratio it never stored. Its accumulator
convention is deliberately not the spec's own s < t indexing: the recorder callback fires AFTER
`step()` has decayed and written sigma for the position, so at callback time `w` holds the sum over
s <= t with exponent gamma^(2(t - s)) while the sigma being read holds the sum over s <= t with
exponent gamma^(t - s). The exponents correspond, which is what makes the ratio meaningful. See
`measure_sigma_structure`'s docstring; do not "correct" it.

Read-only on `runs/` and `data/` apart from the one JSON it writes to `runs/study1/results2/`.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from hbwm.bdh.upstream.bdh import Attention
from hbwm.device import release_memory, select_device
from hbwm.envs import tokenizer as tk
from hbwm.envs.dataset import EpisodeData
from hbwm.instrument.recorder import SigmaRecorder

# `_summary` and `write_concentration` are imported rather than reimplemented so this script's
# percentiles and top-k fractions are the SAME code that produced results2/structure.json; the
# crosscheck below is only meaningful because of that. Nothing in hbwm/ is touched.
from hbwm.instrument.structure import (  # noqa: E402  (grouped apart for the comment above)
    TOP_FRACTIONS,
    _summary,
    participation_ratio,
    write_concentration,
)
from hbwm.probes.eligibility import sample_pairs
from hbwm.probes.run import Study2Config, stratified_subsample
from hbwm.train import load_checkpoint

# ROOT must contain runs/ and data/. Study 1's artifacts live in the sibling worktree, which is the
# default below; override with HBWM_ROOT. OUT_DIR receives the JSON and defaults beside Study 2's
# own aggregated results, where structure.json already lives.
_REPO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("HBWM_ROOT", _REPO / ".claude/worktrees/study1-impl"))
OUT_DIR = Path(os.environ.get("HBWM_POSTHOC_OUT", ROOT / "runs/study1/results2"))
OUT_NAME = "posthoc_cancellation.json"

BDH_RUN = "bdh_g100_lr0.003"
DATA_DIR = ROOT / "data/grid9"
# The four BDH checkpoint-levels Study 2 measured structure on (results2/structure.json keys),
# grouped by checkpoint: seed0 is measured at both levels in ONE set of recorder passes, which is
# exact -- the pair sample does not depend on the level, and only which entry of the payload is
# read differs.
CKPT_LEVELS = {0: [3, 4], 1: [3], 2: [4]}
TINY = 1e-30


def _pearson(x, y):
    """Correlation over the last axis. x, y: [..., N] float64."""
    xc = x - x.mean(dim=-1, keepdim=True)
    yc = y - y.mean(dim=-1, keepdim=True)
    den = (xc.pow(2).sum(-1).sqrt() * yc.pow(2).sum(-1).sqrt()).clamp(min=TINY)
    return (xc * yc).sum(-1) / den


def _rank(x):
    """Ordinal ranks over the last axis (ties broken by position; the write masses are continuous)."""
    return x.argsort(dim=-1).argsort(dim=-1).to(torch.float64)


def study2_pairs(cfg: Study2Config):
    """The `p_tr` that `run_probes_study2` hands to `measure_sigma_structure`.

    The generator's draw order is load-bearing: `sample_pairs` is called on probe_train, probe_val
    and probe_test in that order off the SAME rng before `stratified_subsample` draws from it, so
    reproducing p_tr means reproducing all four calls, not just the first.
    """
    rng = np.random.default_rng(cfg.seed)
    d_tr, d_va, d_te = (EpisodeData(str(DATA_DIR), s) for s in ("probe_train", "probe_val", "probe_test"))
    p_tr_all, _p_va, _p_te = (sample_pairs(d, rng, cfg.per_obj) for d in (d_tr, d_va, d_te))
    return d_tr, stratified_subsample(p_tr_all, cfg.n_train, rng)


def structure_subsample(pairs, n_sample: int, seed: int):
    """Exactly `measure_sigma_structure`'s own subsample: fresh rng(seed), choice, sorted subset."""
    rng = np.random.default_rng(seed)
    take = rng.choice(len(pairs), size=min(n_sample, len(pairs)), replace=False)
    return pairs.subset(np.sort(take))


@torch.no_grad()
def collect(model, data, p, levels, device, batch_eps: int):
    """One set of recorder passes; returns per level (row norms, write mass, write counts, positions).

    Mirrors `measure_sigma_structure`'s pass. Chunked at `batch_eps` episodes because the un-chunked
    form holds a [n_layer, B, nh, N, D] fp32 sigma -- about 12.6 GB at B = 1024 -- for the whole
    pass; this project has already lost a checkpoint to an OOM kill. Chunking is exact, not an
    approximation: `w` is per-episode independent, so each chunk starts from its own zero
    accumulator, and every reported number is a per-example summary.

    Only the row NORMS of sigma are kept, never sigma itself: [M, nh, N] fp32 is 33 MB at M = 1024
    where the full [M, nh, N, D] state would be 2.1 GB, and a is defined from the norms anyway. The
    norms are taken in fp32 with `.norm(dim=-1)` and squared in float64 later, which is bit-for-bit
    what `row_norm_stats` does.
    """
    gamma2 = float(model.hcfg.decay_gamma) ** 2
    obs_pos = tk.obs_positions(data.L)
    nh, N = model.hcfg.n_head, model.hcfg.n_neurons
    acc = {lvl: {"norm": [], "w": [], "nz": [], "pos": []} for lvl in levels}
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
        W = {lvl: torch.zeros(nh, N, len(eps), device=device) for lvl in levels}
        NZ = {lvl: torch.zeros(nh, N, len(eps), device=device) for lvl in levels}

        def fn(pos, payload, by_pos=by_pos, W=W, NZ=NZ):
            for lvl in levels:
                u = payload["x_sparse"][lvl]  # B, nh, N
                q = Attention.rope((float(pos) * model.attn.freqs).view(1, 1, -1), u)
                xn = payload["resid"][lvl].pow(2).sum(-1)  # B
                term = torch.einsum("bhn,b->hnb", q.pow(2), xn)
                W[lvl].mul_(gamma2).add_(term)  # same op order as measure_sigma_structure
                NZ[lvl].add_((term > 0).to(term.dtype))  # steps that actually routed mass into the row
                if pos in by_pos:
                    r = torch.as_tensor(by_pos[pos], device=device)
                    a = acc[lvl]
                    a["norm"].append(payload["sigma"][lvl][r].float().norm(dim=-1).cpu())
                    a["w"].append(W[lvl].permute(2, 0, 1)[r].float().cpu())
                    a["nz"].append(NZ[lvl].permute(2, 0, 1)[r].float().cpu())
                    a["pos"].extend([pos] * len(by_pos[pos]))

        # positions=None on purpose: the write accumulator must advance at EVERY step, so the
        # callback fires each step and stores rows only where `pos in by_pos`.
        rec.run(tokens, None, fn)
        del tokens, W, NZ, fn
        release_memory(device)
    out = {}
    for lvl in levels:
        a = acc[lvl]
        out[lvl] = (torch.cat(a["norm"]), torch.cat(a["w"]), torch.cat(a["nz"]),
                    np.asarray(a["pos"], dtype=np.int64))
        a["norm"].clear(), a["w"].clear(), a["nz"].clear()
    return out


def cancellation_stats(norms, wmass, nz, positions, n_steps: int) -> dict:
    """The cancellation index and its companions. norms/wmass/nz: [M, nh, N] fp32 (M examples)."""
    a = norms.to(torch.float64) ** 2  # = row_norm_stats' `a`
    w = wmass.to(torch.float64)
    counts = nz.to(torch.float64)
    M, nh, N = a.shape
    sum_a, sum_w = a.sum(-1), w.sum(-1)  # [M, nh]

    idx_head = sum_a / sum_w.clamp(min=TINY)  # per (example, head)
    idx_pooled = a.sum((-2, -1)) / w.sum((-2, -1)).clamp(min=TINY)  # per example, heads pooled

    top = {}
    for f in TOP_FRACTIONS:
        k = max(1, int(N * f))  # floor, as in write_concentration
        wv, ii = torch.topk(w, k, dim=-1)
        top[f"index_top_{int(f * 100)}pct"] = _summary(
            torch.gather(a, -1, ii).sum(-1) / wv.sum(-1).clamp(min=TINY))
        top[f"writes_per_row_top_{int(f * 100)}pct"] = _summary(
            torch.gather(counts, -1, ii).mean(-1))

    return {
        "index_pooled": _summary(idx_pooled),
        "index_per_head": _summary(idx_head),
        "index_by_head": {f"h{h}": _summary(idx_head[:, h]) for h in range(nh)},
        **top,
        "corr_w_a_pearson": _summary(_pearson(w, a)),
        "corr_w_a_spearman": _summary(_pearson(_rank(w), _rank(a))),
        "sum_a": _summary(sum_a),
        "sum_w": _summary(sum_w),
        "writes_per_row_median": _summary(counts.median(dim=-1).values),
        "writes_per_row_max": _summary(counts.amax(dim=-1)),
        "frac_rows_never_written": _summary((counts == 0).to(torch.float64).mean(-1)),
        "n_steps_per_episode": int(n_steps),
        "n_steps_before_sampled_pos": _summary(positions + 1),
        "n_examples": int(M), "n_head": int(nh), "n_rows": int(N),
        "all_finite": bool(torch.isfinite(a).all() and torch.isfinite(w).all()),
        "all_positive": bool((sum_a > 0).all() and (sum_w > 0).all()),
        "exploratory": True,
    }


def crosscheck(norms, wmass, ref: dict | None) -> dict:
    """Recompute two numbers structure.json already published, to prove the pass was reproduced."""
    got = {
        "row_norm_participation_ratio":
            _summary(participation_ratio(norms.to(torch.float64) ** 2))["median"],
        "write_participation_ratio":
            write_concentration(wmass)["participation_ratio"]["median"],
    }
    if ref is None:
        return {"reproduced_structure_json": None, **{f"got_{k}": v for k, v in got.items()}}
    want = {"row_norm_participation_ratio": ref["row_norm"]["participation_ratio"]["median"],
            "write_participation_ratio": ref["write_concentration"]["participation_ratio"]["median"]}
    rel = {k: abs(got[k] - want[k]) / max(abs(want[k]), TINY) for k in got}
    return {"reproduced_structure_json": all(v < 1e-6 for v in rel.values()),
            **{f"got_{k}": got[k] for k in got}, **{f"ref_{k}": want[k] for k in want},
            **{f"rel_err_{k}": rel[k] for k in rel}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, nargs="*", default=sorted(CKPT_LEVELS))
    ap.add_argument("--levels", type=int, nargs="*", default=None, help="override the per-seed levels")
    ap.add_argument("--n-sample", type=int, default=Study2Config.structure_n_sample)
    ap.add_argument("--batch-eps", type=int, default=Study2Config.batch_eps)
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
    print(f"device={device} probe_train pairs={len(p_tr)} sampled={len(p)} "
          f"episodes={len(np.unique(p.ep))} batch_eps={args.batch_eps} T={data.T} "
          f"({time.time() - t0:.1f}s to build pairs)")

    ref_path = ROOT / "runs/study1/results2/structure.json"
    refs = json.loads(ref_path.read_text()) if ref_path.exists() else {}
    out, rows = {}, []
    for seed in args.seeds:
        levels = args.levels if args.levels is not None else CKPT_LEVELS[seed]
        ckpt = ROOT / "runs/study1" / BDH_RUN / f"seed{seed}/ckpt.pt"
        model, _, meta = load_checkpoint(ckpt, device)
        t1 = time.time()
        got = collect(model, data, p, levels, device, args.batch_eps)
        secs = time.time() - t1
        for lvl in levels:
            norms, wmass, nz, pos = got[lvl]
            key = f"bdh_g100/seed{seed}/cancellation_L{lvl}"
            r = cancellation_stats(norms, wmass, nz, pos, data.T)
            r["crosscheck"] = crosscheck(norms, wmass,
                                         refs.get(f"bdh_g100/seed{seed}/sigma_structure_L{lvl}")
                                         if args.n_sample == Study2Config.structure_n_sample else None)
            r.update({"level": lvl, "seed": seed, "run": BDH_RUN, "ckpt": str(ckpt),
                      "ckpt_step": int(meta["step"]), "decay_gamma": float(model.hcfg.decay_gamma),
                      "n_sample": args.n_sample, "sample_seed": cfg.seed,
                      "batch_eps": args.batch_eps, "elapsed_s": round(secs, 1)})
            out[key] = r
            rows.append((key, r))
        del got, model
        release_memory(device)
        print(f"  seed{seed} levels={levels}: {secs:.1f}s")

    hdr = (f"{'checkpoint-level':28s} {'index (med [p10,p90])':26s} {'top1%':>7s} {'top10%':>7s} "
           f"{'r(w,a)':>7s} {'rho':>6s} {'wr/row':>7s}")
    print("\n=== cancellation index  sum(a)/sum(w)  (<1 writes cancel, >1 they reinforce) ===")
    print(hdr)
    print("-" * len(hdr))
    for key, r in rows:
        i = r["index_pooled"]
        print(f"{key.replace('bdh_g100/', ''):28s} "
              f"{i['median']:.4f} [{i['p10']:.4f},{i['p90']:.4f}]".ljust(27)
              + f"{r['index_top_1pct']['median']:7.4f} {r['index_top_10pct']['median']:7.4f} "
                f"{r['corr_w_a_pearson']['median']:7.3f} {r['corr_w_a_spearman']['median']:6.3f} "
                f"{r['writes_per_row_median']['median']:7.1f}")
    for key, r in rows:
        c = r["crosscheck"]
        print(f"{key}: sum_a med={r['sum_a']['median']:.4g} sum_w med={r['sum_w']['median']:.4g} "
              f"finite={r['all_finite']} positive={r['all_positive']} "
              f"steps/episode={r['n_steps_per_episode']} "
              f"rows never written={r['frac_rows_never_written']['median']:.4f} "
              f"reproduces structure.json={c['reproduced_structure_json']}")

    if not args.no_write:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Merge by default: one checkpoint-level is ~20 minutes of recorder passes, so a run that
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
