"""Study 2 post-hoc exploratory analysis: does position decodability DEGRADE as an episode runs on?

EXPLORATORY, POST-HOC, NOT PREREGISTERED. Nothing in this script decides anything. No number here
is a criterion, and none of it can support, refute or revise H1 to H8: those decisions stand
exactly as RESULTS.md records them. This is a re-slice of predictions that already exist.

The question. `analysis/posthoc/cancellation_index.py` found that BDH's state sigma is massively
superposed at decay gamma = 1.0: 142 to 214 writes land in each neuron row per episode, and the
cancellation index sum(a)/sum(w) is only 4.9 to 10.7, so roughly 95% of the routed write mass is
destroyed by interference. That is CONSISTENT with interference being why position is not
decodable, but it does not demonstrate it. If accumulating interference is the mechanism, then
decodability should fall as an episode progresses and more writes pile into the same rows.

The test. Study 2's runner saved, for every probe arm, `probes2/<label>_test.npz` with `probs`
(fp16, [n_test, 81]), `label`, `ep`, `t`, `obj` and `bucket`. `t` is the environment step index at
which the pair was queried, so accuracy as a function of `t` needs no model, no checkpoint and no
GPU: it is a pure re-slice of the saved test predictions. We bin the test pairs by `t`, report
accuracy per bin, and summarise the trend with a tie-corrected Spearman correlation between `t` and
per-pair correctness plus an ordinary least squares slope of binned accuracy on bin midpoint.

THE CONFOUND, which this script controls for and reports on either way. `steps_since_seen` is
itself correlated with `t`: an object last seen a long time ago can only be queried late in an
episode, so the `bucket` field (6 levels of steps_since_seen) rises with `t` mechanically. Study 1
and Study 2 already know accuracy falls with `steps_since_seen` (that is H2), so a raw decline
against `t` proves nothing on its own. The load-bearing number is therefore accuracy against `t`
WITHIN each bucket, plus a pooled bucket-fixed-effects slope: a linear probability model
`correct ~ t + bucket dummies`, whose coefficient on `t` is the change in accuracy per environment
step AT FIXED staleness. If accuracy still falls with `t` at fixed bucket, that is evidence for
accumulating interference rather than for mere staleness. If it does not, the superposition story
is WEAKENED and this script says so plainly.

Uncertainty. Test pairs are clustered within episodes, so the Spearman p-value below (a normal
approximation that assumes independent pairs) is descriptive decoration only. The honest interval
is the episode-clustered bootstrap CI on the fixed-effects slope, which resamples episodes with
replacement exactly as `hbwm.probes.probe.bootstrap_ci` does for the preregistered accuracies.

Read-only on `runs/` apart from the one JSON it writes to `runs/study1/results2/`.
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

# BUCKET_NAMES is imported rather than retyped so the bucket indices in this script are the SAME
# 6 levels of steps_since_seen that the preregistered H2 table reports. Nothing in hbwm/ is touched.
from hbwm.probes.eligibility import BUCKET_NAMES

# ROOT must contain runs/. Study 1's artifacts live in the sibling worktree, which is the default
# below; override with HBWM_ROOT. OUT_DIR defaults beside Study 2's own aggregated results.
_REPO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("HBWM_ROOT", _REPO / ".claude/worktrees/study1-impl"))
OUT_DIR = Path(os.environ.get("HBWM_POSTHOC_OUT", ROOT / "runs/study1/results2"))
OUT_NAME = "posthoc_decodability_vs_time.json"

RUNS = ROOT / "runs/study1"
LR_SUFFIX = "_lr0.003"
SEEDS = (0, 1, 2)

# The arms to slice, per model stem. "derot_query_rank_best" is not a label: it is resolved per seed
# to whichever `derot_query_rank_*` arm had the highest validation accuracy, which is the same
# best-val rule `run_probes_study2` uses to pick its H8 spec. `flat_linear` and `mlp_rownorm` are
# the contrast arms; the baselines' `mlp_state` is the control, since lstm and rwkv have no rownorm.
DEROT_BEST = "derot_query_rank_best"
DEFAULT_ARMS = {
    "bdh_g100": [DEROT_BEST, "flat_linear", "mlp_rownorm"],
    "lstm": ["mlp_state"],
    "rwkv": ["mlp_state"],
}
BIN_WIDTH = 8  # L = 96, so 12 bins of 8 environment steps
MIN_BIN_N = 30  # bins thinner than this are reported but held out of the OLS fit
TINY = 1e-30


def val_best(rec: dict) -> float:
    """Study 2's `val_acc` is keyed `"<l2>/<restart>"`, so best val is the max over all entries.

    This is exactly the rule at `run_probes_study2`'s `best_overall`, which is how the reported
    structured readout was chosen in the first place.
    """
    return max(float(v) for v in rec["val_acc"].values())


def resolve_arm(probes_dir: Path, arm: str):
    """(label, json_path, npz_path) for one arm in one seed's probes2/, or None if absent.

    BDH labels carry a `_L<level>` suffix and each seed was measured at its own best level, so the
    level is discovered from the files rather than assumed. A stem with several matching files
    (the derot family, or one seed measured at two levels) is resolved by best validation accuracy.
    """
    pat = "derot_query_rank_*_test.npz" if arm == DEROT_BEST else f"{arm}*_test.npz"
    cands = []
    for npz in sorted(probes_dir.glob(pat)):
        label = npz.name[: -len("_test.npz")]
        if arm != DEROT_BEST and label != arm and not label.startswith(f"{arm}_L"):
            continue  # `flat_linear` must not swallow `flat_linear_bridge`
        js = probes_dir / f"{label}.json"
        if js.exists():
            cands.append((val_best(json.loads(js.read_text())), label, js, npz))
    if not cands:
        return None
    _, label, js, npz = max(cands, key=lambda c: c[0])
    return label, js, npz


def _rankdata(a):
    """Average ranks with tie correction (what Spearman needs; `correct` is 0/1, all ties)."""
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="stable")
    ranks = np.empty(len(a), dtype=np.float64)
    srt = a[order]
    i = 0
    while i < len(srt):
        j = i
        while j + 1 < len(srt) and srt[j + 1] == srt[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def _pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xc, yc = x - x.mean(), y - y.mean()
    den = math.sqrt(float(xc @ xc)) * math.sqrt(float(yc @ yc))
    return float(xc @ yc / den) if den > TINY else float("nan")


def spearman(x, y) -> dict:
    """Tie-corrected Spearman plus a NORMAL-APPROXIMATION p-value that ignores episode clustering.

    The p-value is descriptive only. Test pairs are clustered within episodes (up to 81 pairs per
    episode share one rollout), so the effective sample size is far below `n` and this p is
    anticonservative by construction. The clustered bootstrap on the slope is the real interval.
    """
    n = len(x)
    if n < 3:
        return {"rho": float("nan"), "p_normal_approx": float("nan"), "n": int(n)}
    rho = _pearson(_rankdata(x), _rankdata(y))
    if not math.isfinite(rho) or abs(rho) >= 1.0:
        p = 0.0 if math.isfinite(rho) else float("nan")
    else:
        t = rho * math.sqrt((n - 2) / (1.0 - rho * rho))
        p = math.erfc(abs(t) / math.sqrt(2.0))
    return {"rho": float(rho), "p_normal_approx": float(p), "n": int(n)}


def bin_index(t, width: int):
    return (np.asarray(t, dtype=np.int64) - 1) // width  # t is 1-based over 1..L


def binned_accuracy(t, correct, width: int) -> list:
    """Per-bin accuracy and count, in bin order. Midpoint is nominal; `t_mean` is what is there."""
    b = bin_index(t, width)
    out = []
    for k in range(int(b.max()) + 1):
        m = b == k
        n = int(m.sum())
        out.append({
            "bin": k,
            "t_lo": int(width * k + 1),
            "t_hi": int(width * (k + 1)),
            "midpoint": float(width * k + (width + 1) / 2.0),
            "t_mean": float(t[m].mean()) if n else float("nan"),
            "n": n,
            "acc": float(correct[m].mean()) if n else float("nan"),
        })
    return out


def ols_slope(bins, min_n: int) -> dict:
    """Unweighted OLS of binned accuracy on bin midpoint, over bins with at least `min_n` pairs."""
    use = [b for b in bins if b["n"] >= min_n and math.isfinite(b["acc"])]
    if len(use) < 3:
        return {"slope_per_step": float("nan"), "intercept": float("nan"), "r2": float("nan"),
                "n_bins_used": len(use), "n_bins_total": len(bins), "delta_over_range": float("nan")}
    x = np.array([b["midpoint"] for b in use], dtype=np.float64)
    y = np.array([b["acc"] for b in use], dtype=np.float64)
    A = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "slope_per_step": float(coef[1]),
        "intercept": float(coef[0]),
        "r2": float(1.0 - float((resid ** 2).sum()) / ss_tot) if ss_tot > TINY else float("nan"),
        "n_bins_used": len(use),
        "n_bins_total": len(bins),
        # accuracy change implied across the span of bins actually fitted
        "delta_over_range": float(coef[1] * (x.max() - x.min())),
    }


def _design(t, bucket, n_buckets: int):
    """[1, t, bucket dummies for levels 1..n_buckets-1]. Level 0 is the reference cell."""
    cols = [np.ones(len(t), dtype=np.float64), np.asarray(t, dtype=np.float64)]
    for b in range(1, n_buckets):
        cols.append((np.asarray(bucket) == b).astype(np.float64))
    return np.column_stack(cols)


def _lpm_slope(X, y) -> float:
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(coef[1])


def fixed_effects_slope(t, correct, bucket, ep, n_boot: int, seed: int) -> dict:
    """Coefficient on `t` in `correct ~ t + bucket dummies`: accuracy per step AT FIXED staleness.

    This is THE number the confound argument turns on. A linear probability model is used rather
    than a logit because the quantity of interest is a change in accuracy (the same scale as every
    other number in this study) and because the bucket dummies absorb the staleness gradient
    without any distributional assumption. The CI is a bootstrap over EPISODES, not over pairs:
    `bootstrap_ci` in `hbwm/probes/probe.py` clusters the preregistered accuracies the same way,
    because pairs inside one episode share a rollout and are not independent.
    """
    n_buckets = len(BUCKET_NAMES)
    y = np.asarray(correct, dtype=np.float64)
    X = _design(t, bucket, n_buckets)
    point = _lpm_slope(X, y)
    raw = _lpm_slope(np.column_stack([np.ones(len(y)), np.asarray(t, dtype=np.float64)]), y)
    boots = []
    if n_boot > 0:
        rng = np.random.default_rng(seed)
        eps = np.unique(ep)
        rows_of = {int(e): np.where(ep == e)[0] for e in eps}
        for _ in range(n_boot):
            pick = rng.choice(eps, size=len(eps), replace=True)
            idx = np.concatenate([rows_of[int(e)] for e in pick])
            try:
                boots.append(_lpm_slope(X[idx], y[idx]))
            except np.linalg.LinAlgError:  # a resample can drop a whole bucket level
                continue
    ci = ([float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]
          if len(boots) >= 20 else [float("nan"), float("nan")])
    return {
        "slope_per_step": float(point),
        "slope_per_step_no_control": float(raw),
        "ci95_cluster_ep": ci,
        "n_boot": int(len(boots)),
        "n_episodes": int(len(np.unique(ep))),
        "excludes_zero": bool(math.isfinite(ci[0]) and (ci[0] > 0 or ci[1] < 0)),
    }


def analyse_arm(npz_path: Path, meta: dict, width: int, min_bin_n: int, n_boot: int,
                seed: int) -> dict:
    d = np.load(npz_path)
    t, bucket, ep = d["t"], d["bucket"], d["ep"]
    correct = (np.asarray(d["probs"]).astype(np.float32).argmax(1) == d["label"]).astype(np.float64)
    del d
    overall = binned_accuracy(t, correct, width)
    per_bucket = {}
    for b in range(len(BUCKET_NAMES)):
        m = bucket == b
        if not m.any():
            continue
        bins = binned_accuracy(t[m], correct[m], width)
        per_bucket[BUCKET_NAMES[b]] = {
            "bucket_index": b,
            "n": int(m.sum()),
            "acc": float(correct[m].mean()),
            "t_min": int(t[m].min()),
            "t_max": int(t[m].max()),
            "bins": bins,
            "spearman_t_correct": spearman(t[m], correct[m]),
            "ols": ols_slope(bins, min_bin_n),
        }
    return {
        **meta,
        "n_test": int(len(t)),
        "acc_overall": float(correct.mean()),
        "t_min": int(t.min()),
        "t_max": int(t.max()),
        "bin_width": int(width),
        "bins": overall,
        "spearman_t_correct": spearman(t, correct),
        "ols": ols_slope(overall, min_bin_n),
        # documents the confound itself: how strongly staleness rises with t in this test set
        "spearman_t_bucket": spearman(t, bucket),
        "within_bucket": per_bucket,
        "fixed_effects": fixed_effects_slope(t, correct, bucket, ep, n_boot, seed),
        "exploratory": True,
        "preregistered": False,
    }


def pooled(records: list, path: str) -> dict:
    """Mean and std ACROSS SEEDS of the per-seed statistics, never a pool of raw rows.

    Seeds are different checkpoints, so concatenating their test pairs would mix three separate
    models into one regression. Every preregistered table in this study aggregates the same way.
    """
    def grab(fn):
        v = [fn(r) for r in records]
        v = [x for x in v if math.isfinite(x)]
        return {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                "values": [float(x) for x in v], "n_seeds": len(v)}
    return {
        "path": path,
        "acc_overall": grab(lambda r: r["acc_overall"]),
        "spearman_rho": grab(lambda r: r["spearman_t_correct"]["rho"]),
        "ols_slope_per_step": grab(lambda r: r["ols"]["slope_per_step"]),
        "fe_slope_per_step": grab(lambda r: r["fixed_effects"]["slope_per_step"]),
        "fe_slope_no_control": grab(lambda r: r["fixed_effects"]["slope_per_step_no_control"]),
        "n_seeds_signed_ci_excludes_zero": int(sum(r["fixed_effects"]["excludes_zero"]
                                                   for r in records)),
        "n_seeds_fe_negative": int(sum(r["fixed_effects"]["slope_per_step"] < 0 for r in records)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stems", nargs="*", default=list(DEFAULT_ARMS))
    ap.add_argument("--arms", nargs="*", default=None, help="override the per-stem arm list")
    ap.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    ap.add_argument("--bin-width", type=int, default=BIN_WIDTH)
    ap.add_argument("--min-bin-n", type=int, default=MIN_BIN_N)
    ap.add_argument("--n-boot", type=int, default=1000, help="episode-clustered bootstrap draws")
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT_DIR / OUT_NAME))
    ap.add_argument("--no-write", action="store_true", help="print only; for the validation slice")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace the output JSON instead of merging this run's keys into it")
    args = ap.parse_args()

    out, rows, by_path = {}, [], {}
    for stem in args.stems:
        arms = args.arms if args.arms is not None else DEFAULT_ARMS[stem]
        for arm in arms:
            for seed in args.seeds:
                pdir = RUNS / f"{stem}{LR_SUFFIX}" / f"seed{seed}" / "probes2"
                got = resolve_arm(pdir, arm) if pdir.exists() else None
                if got is None:
                    print(f"  skip {stem}/seed{seed}/{arm}: no matching probes2 arm under {pdir}")
                    continue
                label, js, npz = got
                rec = json.loads(js.read_text())
                meta = {"stem": stem, "seed": seed, "arm": arm, "label": label,
                        "level": rec.get("level"), "family": rec.get("family"),
                        "test_acc_reported": rec.get("test_acc"), "chance": rec.get("chance"),
                        "val_acc_best": val_best(rec), "npz": str(npz)}
                r = analyse_arm(npz, meta, args.bin_width, args.min_bin_n, args.n_boot,
                                args.boot_seed + seed)
                key = f"{stem}/seed{seed}/{arm}"
                out[key] = r
                rows.append((key, r))
                by_path.setdefault(f"{stem}/{arm}", []).append(r)
                acc_chk = abs(r["acc_overall"] - (rec.get("test_acc") or r["acc_overall"]))
                print(f"  {key:38s} label={label:24s} n={r['n_test']:6d} "
                      f"acc={r['acc_overall']:.4f} (json delta {acc_chk:.2e})")

    summ = {f"summary/{k}": pooled(v, k) for k, v in by_path.items() if v}
    out.update(summ)

    hdr = (f"{'stem/seed/arm':40s} {'acc':>6s} {'rho(t,corr)':>11s} {'p':>9s} "
           f"{'OLS/step':>9s} {'FE/step':>9s} {'FE ci95':>22s} {'FE raw':>9s}")
    print("\n=== accuracy vs environment step t (FE = at fixed steps_since_seen bucket) ===")
    print(hdr)
    print("-" * len(hdr))
    for key, r in rows:
        sp, fe = r["spearman_t_correct"], r["fixed_effects"]
        ci = f"[{fe['ci95_cluster_ep'][0]:+.2e},{fe['ci95_cluster_ep'][1]:+.2e}]"
        print(f"{key:40s} {r['acc_overall']:6.4f} {sp['rho']:+11.4f} "
              f"{sp['p_normal_approx']:9.2e} {r['ols']['slope_per_step']:+9.2e} "
              f"{fe['slope_per_step']:+9.2e} {ci:>22s} "
              f"{fe['slope_per_step_no_control']:+9.2e}")

    print("\n=== per-bin accuracy (all buckets pooled) ===")
    for key, r in rows:
        cells = " ".join(f"{b['acc']:.3f}({b['n']})" for b in r["bins"])
        print(f"{key}: t {r['t_min']}..{r['t_max']}  {cells}")

    print("\n=== accuracy vs t WITHIN each steps_since_seen bucket (the confound control) ===")
    for key, r in rows:
        print(f"{key}  [spearman(t,bucket)={r['spearman_t_bucket']['rho']:+.3f}]")
        for name, b in r["within_bucket"].items():
            cells = " ".join(f"{x['acc']:.3f}({x['n']})" for x in b["bins"] if x["n"])
            print(f"    bucket {name:>5s} n={b['n']:6d} acc={b['acc']:.4f} "
                  f"rho={b['spearman_t_correct']['rho']:+.4f} "
                  f"ols/step={b['ols']['slope_per_step']:+.2e} | {cells}")

    print("\n=== pooled across seeds ===")
    for k, s in summ.items():
        print(f"{k}: acc {s['acc_overall']['mean']:.4f} +- {s['acc_overall']['std']:.4f} | "
              f"rho {s['spearman_rho']['mean']:+.4f} +- {s['spearman_rho']['std']:.4f} | "
              f"OLS/step {s['ols_slope_per_step']['mean']:+.3e} | "
              f"FE/step {s['fe_slope_per_step']['mean']:+.3e} "
              f"+- {s['fe_slope_per_step']['std']:.1e} "
              f"(no control {s['fe_slope_no_control']['mean']:+.3e}); "
              f"{s['n_seeds_fe_negative']}/{s['acc_overall']['n_seeds']} FE slopes negative, "
              f"{s['n_seeds_signed_ci_excludes_zero']}/{s['acc_overall']['n_seeds']} "
              f"CIs exclude zero")

    print("\nReading guide. A negative FE/step whose episode-clustered CI excludes zero, on the BDH "
          "sigma arms and not on the lstm/rwkv controls, is evidence that accumulating interference "
          "and not mere staleness is what erodes decodability. An FE/step at or above zero, or one "
          "whose CI covers zero, WEAKENS the superposition story: the cancellation index would then "
          "be consistent with undecodability without being demonstrated as its cause. Neither "
          "outcome revises H1 to H8.")

    if not args.no_write:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        merged = {}
        if path.exists() and not args.overwrite:
            merged = json.loads(path.read_text())
        merged.update(out)
        path.write_text(json.dumps(merged, indent=2) + "\n")
        print(f"\nwrote {path} ({len(merged)} keys)")


if __name__ == "__main__":
    main()
