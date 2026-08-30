"""EXPLORATORY post-hoc probe of `yKV`, BDH's own associative read of sigma.

NOT PREREGISTERED. This script was written after Study 1 and Study 2 closed and it decides
nothing: it cannot revise H5, H6, H7 or H8, cannot reopen a preregistered threshold, and cannot
promote its own numbers to a finding. H1-H4 (Study 1) and H5-H8 (Study 2) stand exactly as
recorded in RESULTS.md regardless of what comes out of here.

The question. Every probe family run so far read the raw state sigma [n_head, n_neurons, n_embd]
with a *static, learned* query. The architecture itself reads sigma with a query computed from the
current token: q = rope(relu(x_t @ W_enc), t), then yKV[h,d] = sum_n q[h,n] * sigma[h,n,d]. `yKV`
is therefore what the circuit actually extracts on this timestep, and it has never been probed:
Study 1's BDH_FEATURES are only ("sigma_full", "sigma_rownorm", "x_sparse", "resid"). It is also
tiny -- n_head * n_embd = 256 dims instead of sigma's 524,288 -- so a plain linear readout on it is
cheap and well-conditioned.

Protocol mirrors Study 2 so the numbers are comparable: splits probe_train / probe_val /
probe_test, per_obj=8, 24,000 stratified training pairs, probe seed 0, L2 grid
[1e-4, 1e-3, 1e-2, 1e-1], 20 epochs, lr 1e-3, batch 512, selection on probe_val, reporting on
probe_test, all 6 levels, model seeds 0/1/2.

Read-only on runs/ and data/ except for the single JSON it writes to
runs/study1/results2/posthoc_ykv_probe.json. It imports the unmodified probe machinery
(hbwm.probes.eligibility, hbwm.probes.probe, hbwm.probes.run.stratified_subsample,
hbwm.instrument.recorder) and adds only its own extraction loop, which mirrors the grouping
hbwm/probes/extract.py:iter_features does.

Usage:
    uv run python analysis/posthoc/ykv_probe.py --validate   # cheap smoke check, writes nothing
    uv run python analysis/posthoc/ykv_probe.py              # full sweep, writes the JSON
"""

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from hbwm.device import release_memory, select_device
from hbwm.envs import tokenizer as tk
from hbwm.envs.dataset import EpisodeData
from hbwm.instrument.recorder import SigmaRecorder
from hbwm.probes.eligibility import sample_pairs
from hbwm.probes.probe import (
    accuracy,
    bootstrap_ci,
    majority_chance,
    predict_proba,
    train_probes_multi,
)
from hbwm.probes.run import stratified_subsample
from hbwm.train import load_checkpoint

# ROOT must contain runs/ and data/. Study 1's artifacts live in the sibling worktree, which is the
# default below; override with HBWM_ROOT to point at any other checkout of the same layout.
_REPO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("HBWM_ROOT", _REPO / ".claude/worktrees/study1-impl"))
RUN = "bdh_g100_lr0.003"
DATA_DIR = ROOT / "data/grid9"
OUT_PATH = ROOT / "runs/study1/results2/posthoc_ykv_probe.json"

# Study 2's preregistered probe hyperparameters (hbwm/probes/run.py:Study2Config defaults).
L2_GRID = [1e-4, 1e-3, 1e-2, 1e-1]
EPOCHS, LR, BATCH, PER_OBJ, N_TRAIN, PROBE_SEED = 20, 1e-3, 512, 8, 24000, 0

# Published reference points, for context only. Nothing here is compared against a threshold.
REFERENCE = {
    "bdh_sigma_full_flat_linear": 0.101,
    "bdh_best_structured_derot_query_rank_r": 0.159,
    "bdh_mlp_rownorm": 0.144,
    "lstm_state_mlp_state": 0.146,
    "rwkv_state_mlp_state": 0.165,
    "majority_chance_published": 0.011,
    "oracle_ceiling_published": 1.000,
}


# ---------------------------------------------------------------- extraction

def _batch_ykv(rec, tokens_np, by_pos, levels, device, out, filled):
    """One recorder pass over a batch of episodes, writing yKV rows into `out[level]`.

    A frame of its own so the device token tensor and the closure die on return (same reason
    hbwm/probes/extract.py:_batch_features is factored out). The payload aliases live state that
    the next step() mutates in place, so every level is indexed and copied to CPU *inside* fn.
    """
    tokens = torch.from_numpy(tokens_np.astype(np.int64)).to(device)

    def fn(pos, payload):
        items = by_pos[pos]
        rows = torch.as_tensor([r for _, r in items], device=device)
        idx = np.array([i for i, _ in items], dtype=np.int64)
        for lv in levels:
            y = payload["yKV"][lv]  # [B, n_head, n_embd], aliases live state
            out[lv][idx] = y.reshape(y.size(0), -1)[rows].float().cpu().numpy()
        filled[idx] = True

    rec.run(tokens, sorted(by_pos), fn)


def extract_ykv(model, data, pairs, levels, device, batch_eps=32, release_every=4, log=None,
                cache=None):
    """{level: X[len(pairs), n_head*n_embd]} for the sampled (episode, t) positions.

    All requested levels come out of a single recorder pass, since step() returns every level's
    yKV at once. Pairs are grouped by token position exactly as iter_features does.

    `cache` is an optional .npz path. Extraction is the whole cost of this script (~1 GPU-hour per
    seed) and the job is long enough to be interrupted, so a completed (seed, split) is written out
    and reused on the next invocation. The cache is keyed on the pair count and levels; anything
    that does not match is recomputed. Deleting the cache directory only costs time.
    """
    c = model.hcfg
    feat_dim = c.n_head * c.n_embd
    n = len(pairs)
    if cache is not None and Path(cache).exists():
        z = np.load(cache)
        if all(f"L{lv}" in z.files for lv in levels) and z[f"L{levels[0]}"].shape == (n, feat_dim):
            print(f"    [{log}] cache hit {cache}", flush=True)
            return {lv: z[f"L{lv}"] for lv in levels}
        print(f"    [{log}] cache miss (shape/levels changed), recomputing", flush=True)
    out = {lv: np.empty((n, feat_dim), dtype=np.float32) for lv in levels}
    filled = np.zeros(n, dtype=bool)
    rec = SigmaRecorder(model)
    obs_pos = tk.obs_positions(data.L)
    unique_eps = np.unique(pairs.ep)
    n_batches = (len(unique_eps) + batch_eps - 1) // batch_eps
    t0 = time.time()
    for b, b0 in enumerate(range(0, len(unique_eps), batch_eps), start=1):
        eps = unique_eps[b0 : b0 + batch_eps]
        row_of_ep = {int(e): i for i, e in enumerate(eps)}
        idx_b = np.where(np.isin(pairs.ep, eps))[0]
        by_pos = defaultdict(list)
        for i in idx_b:
            by_pos[int(obs_pos[pairs.t[i]])].append((int(i), row_of_ep[int(pairs.ep[i])]))
        _batch_ykv(rec, data.tokens[eps], by_pos, levels, device, out, filled)
        if b % release_every == 0:
            release_memory(device)
        if log and (b == 1 or b % 10 == 0 or b == n_batches):
            print(f"    [{log}] batch {b}/{n_batches}  {time.time() - t0:.0f}s", flush=True)
    release_memory(device)
    assert filled.all(), f"{int((~filled).sum())} of {n} pairs never filled"
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(cache).with_suffix(".tmp.npz")  # atomic: a kill mid-write must not leave a
        np.savez(tmp, **{f"L{lv}": out[lv] for lv in levels})  # half-written file to be "hit" later
        tmp.replace(cache)
    return out


# ---------------------------------------------------------------- protocol

def build_pairs(d_tr, d_va, d_te, per_obj, n_train, seed):
    """Study 2's pair construction, in its exact rng order (hbwm/probes/run.py:run_probes_study2)."""
    rng = np.random.default_rng(seed)
    p_tr_all, p_va, p_te = (sample_pairs(d, rng, per_obj) for d in (d_tr, d_va, d_te))
    p_tr = stratified_subsample(p_tr_all, n_train, rng)
    return p_tr, p_va, p_te


def limit_eps(pairs, max_eps):
    """Validation-only: keep pairs from the first `max_eps` episodes so a smoke run is one batch."""
    if not max_eps:
        return pairs
    keep = np.unique(pairs.ep)[:max_eps]
    return pairs.subset(np.where(np.isin(pairs.ep, keep))[0])


def fit_level(X_tr, y_tr, X_va, y_va, X_te, y_te, n_classes, args, device):
    """Train the L2 grid jointly, select on val, report the winner on test."""
    probes = train_probes_multi(X_tr, y_tr, n_classes, L2_GRID, args.epochs, LR, BATCH,
                                PROBE_SEED, device)
    val = {l2: accuracy(predict_proba(p, X_va, device=device), y_va) for l2, p in probes.items()}
    best_l2 = max(val, key=lambda k: (val[k], -k))
    te_probs = predict_proba(probes[best_l2], X_te, device=device)
    correct = (te_probs.argmax(1) == np.asarray(y_te)).astype(np.float64)
    return {
        "val_acc": {f"{k:g}": v for k, v in val.items()},
        "best_l2": best_l2,
        "val_acc_best": val[best_l2],
        "test_acc": float(correct.mean()),
        "_correct": correct,
    }


def run_seed(seed, levels, pairs, splits, args, device):
    p_tr, p_va, p_te = pairs
    d_tr, d_va, d_te = splits
    ckpt = ROOT / "runs/study1" / RUN / f"seed{seed}" / "ckpt.pt"
    model, cfg, meta = load_checkpoint(ckpt, device)
    c = model.hcfg
    print(f"  seed{seed}: step={meta['step']} val_ce={meta['val_ce']:.4f} "
          f"n_head={c.n_head} n_embd={c.n_embd} feat_dim={c.n_head * c.n_embd}", flush=True)
    X = {}
    for name, d, p in (("train", d_tr, p_tr), ("val", d_va, p_va), ("test", d_te, p_te)):
        cache = None if not args.cache_dir else Path(args.cache_dir) / f"s{seed}_{name}_{len(p)}.npz"
        X[name] = extract_ykv(model, d, p, levels, device, args.batch_eps, log=f"s{seed}/{name}",
                              cache=cache)
    n_classes = d_tr.G * d_tr.G
    out = {}
    for lv in levels:
        r = fit_level(X["train"][lv], p_tr.label, X["val"][lv], p_va.label,
                      X["test"][lv], p_te.label, n_classes, args, device)
        out[lv] = r
        print(f"    L{lv}: val={r['val_acc_best']:.4f} (l2={r['best_l2']:g}) "
              f"test={r['test_acc']:.4f}", flush=True)
    del X, model
    release_memory(device)
    return out


# ---------------------------------------------------------------- reporting

def summarize(per_seed, p_te, n_boot):
    """Best level per seed selected on probe_val, its test accuracy and episode-bootstrap CI."""
    best = {}
    for seed, res in per_seed.items():
        lv = max(res, key=lambda k: (res[k]["val_acc_best"], -k))
        lo, hi = bootstrap_ci(res[lv]["_correct"], p_te.ep, n_boot=n_boot)
        best[seed] = {"level": lv, "val_acc": res[lv]["val_acc_best"], "best_l2": res[lv]["best_l2"],
                      "test_acc": res[lv]["test_acc"], "test_ci95": [lo, hi]}
    accs = [b["test_acc"] for b in best.values()]
    return best, float(np.mean(accs)), float(np.std(accs))


def print_table(per_seed, best, mean_acc, std_acc, chance, ceiling, n_boot):
    seeds = sorted(per_seed)
    levels = sorted(per_seed[seeds[0]])
    print("\n=== yKV linear probe, object cell id (EXPLORATORY, not preregistered) ===")
    head = "level | " + " | ".join(f"s{s} val   test " for s in seeds) + " | mean test"
    print(head)
    print("-" * len(head))
    for lv in levels:
        cells = " | ".join(f"{per_seed[s][lv]['val_acc_best']:.4f} {per_seed[s][lv]['test_acc']:.4f}"
                           for s in seeds)
        m = np.mean([per_seed[s][lv]["test_acc"] for s in seeds])
        print(f"  L{lv}  | {cells} |    {m:.4f}")
    print("-" * len(head))
    for s in seeds:
        b = best[s]
        print(f"  best level seed{s}: L{b['level']} (val {b['val_acc']:.4f}, l2={b['best_l2']:g}) "
              f"-> test {b['test_acc']:.4f}  95% CI [{b['test_ci95'][0]:.4f}, {b['test_ci95'][1]:.4f}] "
              f"({n_boot} episode bootstraps)")
    print(f"\n  yKV best-level mean over seeds: {mean_acc:.4f} (sd {std_acc:.4f})")
    print(f"  majority chance: {chance:.4f}   oracle ceiling: {ceiling:.4f}")
    print("\n  Reference points (published, for context only):")
    for k, v in REFERENCE.items():
        delta = mean_acc - v
        print(f"    {k:<42s} {v:.3f}   (yKV {delta:+.3f})")
    print("\n  EXPLORATORY / post-hoc. Decides nothing; cannot revise H5-H8.")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--levels", default="0,1,2,3,4,5")
    ap.add_argument("--n-train", type=int, default=N_TRAIN)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch-eps", type=int, default=32)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--max-eps", type=int, default=0,
                    help="validation only: keep pairs from the first N episodes of each split")
    ap.add_argument("--device", default=None)
    ap.add_argument("--cache-dir", default="/tmp/ykv_probe_cache",
                    help="reuse extracted yKV across restarts; '' disables (scratch, not an output)")
    ap.add_argument("--out", default=str(OUT_PATH))
    ap.add_argument("--validate", action="store_true",
                    help="cheap smoke check: seed 0, level 3, few pairs, 2 epochs; writes no JSON")
    args = ap.parse_args()
    if args.validate:
        # One seed, one level, ~3 recorder batches per split, a few hundred training pairs.
        # Epochs stay at the protocol's 20: at 2 epochs the probe is still under-optimized and
        # lands *below* majority chance, which makes the "above chance" check meaningless. The
        # extra 18 epochs on ~800 rows cost seconds; the recorder passes dominate the runtime.
        args.seeds, args.levels = "0", "3"
        args.max_eps, args.n_boot, args.cache_dir = 96, 200, ""
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    levels = sorted({int(x) for x in args.levels.split(",") if x != ""})

    device = select_device(args.device)
    t0 = time.time()
    print(f"root={ROOT} device={device} seeds={seeds} levels={levels} "
          f"n_train={args.n_train} epochs={args.epochs} validate={args.validate}", flush=True)
    d_tr, d_va, d_te = (EpisodeData(str(DATA_DIR), s) for s in ("probe_train", "probe_val", "probe_test"))
    p_tr, p_va, p_te = build_pairs(d_tr, d_va, d_te, PER_OBJ, args.n_train, PROBE_SEED)
    p_tr, p_va, p_te = (limit_eps(p, args.max_eps) for p in (p_tr, p_va, p_te))
    print(f"pairs: train={len(p_tr)} val={len(p_va)} test={len(p_te)} "
          f"(episodes {len(np.unique(p_tr.ep))}/{len(np.unique(p_va.ep))}/{len(np.unique(p_te.ep))})",
          flush=True)
    chance = majority_chance(p_tr.label, p_te.label)
    ceiling = float((p_te.oracle == p_te.label).mean())

    per_seed = {s: run_seed(s, levels, (p_tr, p_va, p_te), (d_tr, d_va, d_te), args, device)
                for s in seeds}
    best, mean_acc, std_acc = summarize(per_seed, p_te, args.n_boot)
    print_table(per_seed, best, mean_acc, std_acc, chance, ceiling, args.n_boot)

    payload = {
        "analysis": "posthoc_ykv_probe",
        "status": "EXPLORATORY, post-hoc, NOT preregistered; decides nothing and cannot revise H5-H8",
        "feature": "yKV = sum_n rope(relu(x_t @ W_enc), t)[h,n] * sigma[h,n,d], flattened over "
                   "(n_head, n_embd); the model's own associative read of sigma",
        "probe": "flat linear (multinomial) on standardized yKV",
        "run": RUN,
        "seeds": seeds,
        "levels": levels,
        "protocol": {"splits": ["probe_train", "probe_val", "probe_test"], "per_obj": PER_OBJ,
                     "n_train": args.n_train, "l2_grid": L2_GRID, "epochs": args.epochs, "lr": LR,
                     "batch": BATCH, "probe_seed": PROBE_SEED, "select_on": "probe_val",
                     "report_on": "probe_test", "n_boot": args.n_boot, "batch_eps": args.batch_eps,
                     "max_eps": args.max_eps},
        "n_pairs": {"train": len(p_tr), "val": len(p_va), "test": len(p_te)},
        "n_classes": d_tr.G * d_tr.G,
        "chance": chance,
        "ceiling": ceiling,
        "per_seed": {str(s): {f"L{lv}": {k: v for k, v in r.items() if not k.startswith("_")}
                              for lv, r in res.items()} for s, res in per_seed.items()},
        "per_level_mean_test": {f"L{lv}": float(np.mean([per_seed[s][lv]["test_acc"] for s in seeds]))
                                for lv in levels},
        "best_per_seed": {str(s): b for s, b in best.items()},
        "headline": {"ykv_best_level_mean_test_acc": mean_acc, "sd_over_seeds": std_acc},
        "reference": REFERENCE,
        "elapsed_s": round(time.time() - t0, 1),
    }
    if args.validate:
        print("\n[validate] no JSON written")
    else:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
