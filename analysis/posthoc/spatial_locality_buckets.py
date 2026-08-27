"""
Study 1 post-hoc exploratory follow-up (extension of spatial_locality.py):

H2 asked whether exact-cell probe accuracy decays gracefully with steps-since-seen.
For BDH gamma=1 the curve was flat and low (~0.08-0.12 across all six buckets) and
the write-up read that as "flatness of a weak signal, not robust memory". But
spatial_locality.py showed BDH's errors are spatially local (within-radius-1 accuracy
3.2-5.6x its own chance rate overall) -- so exact-match may be floor-limited, hiding
a real decaying memory signal that a graded (distance-aware) metric can see.

This script:
  1. Reproduces the H2 exact-accuracy-by-bucket curve as a correctness check against
     RESULTS.md.
  2. Recomputes within-radius-1 accuracy, mean Chebyshev error, and expected distance
     under the full predictive distribution, all broken down by the same six
     steps-since-seen buckets, for every spec already analyzed by spatial_locality.py.
  3. Applies Study 1's own H2 "graceful decay" shape test (acc(33-64) >= 0.5*acc(1-4),
     no bucket below 50% of its predecessor) to the graded within-r1-accuracy curve,
     labeled explicitly as an EXPLORATORY, non-preregistered recomputation.
  4. Builds a synthetic "blurred one-hot" reference (Gaussian blur of the true cell at
     sigma_blur in {1, 2, 3} cells) run through the identical metric pipeline, to give
     a calibrated answer to "how coarse is the observed code, in cells".

Reads only already-saved probe .npz outputs under runs/study1/ and the probe_test
split under data/grid9/. No model inference. Post-hoc / exploratory, NOT the
preregistered H2.
"""
import json
import os
from pathlib import Path

import numpy as np

# ROOT must contain runs/, data/ and an importable hbwm/ (h2_curve and BUCKET_NAMES are imported
# from it below, so this path is also prepended to sys.path). Study 1's artifacts live in the
# sibling worktree, which is the default; override with HBWM_ROOT. OUT_DIR receives the JSON dump.
_REPO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("HBWM_ROOT", _REPO / ".claude/worktrees/study1-impl"))
OUT_DIR = Path(os.environ.get("HBWM_POSTHOC_OUT", Path.cwd()))

import sys
sys.path.insert(0, str(ROOT))
from hbwm.probes.decisions import h2_curve  # noqa: E402
from hbwm.probes.eligibility import BUCKET_NAMES  # noqa: E402

G = 9
N_CLASSES = G * G
SEEDS = [0, 1, 2]

# ---- per-seed spec selection (identical to spatial_locality.py) ----
BDH_RUN = "bdh_g100_lr0.003"
SIGMA_FULL_LEVELS = {0: 3, 1: 3, 2: 4}
SIGMA_ROWNORM_LEVELS = {0: 4, 1: 3, 2: 4}

SPECS = {
    "BDH sigma_full (best L/seed)": [
        (ROOT / "runs/study1" / BDH_RUN / f"seed{s}/probes/sigma_full_L{SIGMA_FULL_LEVELS[s]}_test.npz", s)
        for s in SEEDS
    ],
    "BDH sigma_rownorm (best L/seed)": [
        (ROOT / "runs/study1" / BDH_RUN / f"seed{s}/probes/sigma_rownorm_L{SIGMA_ROWNORM_LEVELS[s]}_test.npz", s)
        for s in SEEDS
    ],
    "LSTM state_vec": [
        (ROOT / "runs/study1/lstm_lr0.003" / f"seed{s}/probes/state_vec_test.npz", s)
        for s in SEEDS
    ],
    "RWKV state_vec": [
        (ROOT / "runs/study1/rwkv_lr0.003" / f"seed{s}/probes/state_vec_test.npz", s)
        for s in SEEDS
    ],
}

# ---------------------------------------------------------------------------
# Distance machinery over the 81-cell grid (cell = y*G + x) -- identical to
# spatial_locality.py's per-row chance computation.
# ---------------------------------------------------------------------------
_c = np.arange(N_CLASSES)
_X = _c % G
_Y = _c // G
DX = np.abs(_X[:, None] - _X[None, :])
DY = np.abs(_Y[:, None] - _Y[None, :])
DIST_CHEB = np.maximum(DX, DY).astype(np.float64)   # [81,81]
DIST_EUC = np.sqrt((DX.astype(np.float64) ** 2) + (DY.astype(np.float64) ** 2))  # [81,81]

NEIGH_R1_COUNT = (DIST_CHEB <= 1).sum(axis=1)  # per true-cell count of cells within radius 1 (incl. self)
ROW_MEAN_DIST_CHEB = DIST_CHEB.mean(axis=1)    # E[dist] under a uniform predictive dist, per true cell

BUCKET_EDGES_DISPLAY = ["1-4", "5-8", "9-16", "17-32", "33-64", "65+"]
assert BUCKET_EDGES_DISPLAY == BUCKET_NAMES


def per_row_data(npz_path: Path):
    d = np.load(npz_path)
    probs = d["probs"].astype(np.float32)
    label = d["label"].astype(np.int64)
    bucket = d["bucket"].astype(np.int64)
    pred = probs.argmax(axis=1)
    cheb = DIST_CHEB[pred, label]
    within_r1 = (cheb <= 1).astype(np.float64)
    chance_r1 = NEIGH_R1_COUNT[label] / N_CLASSES
    exact = (pred == label).astype(np.float64)
    dist_to_label_cheb = DIST_CHEB[:, label]  # [81, n]
    exp_d_cheb = np.einsum("nc,cn->n", probs, dist_to_label_cheb)
    uniform_null = ROW_MEAN_DIST_CHEB[label]
    return {
        "bucket": bucket, "exact": exact, "cheb": cheb, "within_r1": within_r1,
        "chance_r1": chance_r1, "exp_d_cheb": exp_d_cheb, "uniform_null": uniform_null,
    }


def bucket_breakdown(rd: dict) -> dict:
    """Per-bucket means for one seed's spec."""
    out = {}
    for b, name in enumerate(BUCKET_NAMES):
        m = rd["bucket"] == b
        n = int(m.sum())
        if n == 0:
            out[name] = None
            continue
        out[name] = {
            "n": n,
            "exact_acc": float(rd["exact"][m].mean()),
            "within_r1_acc": float(rd["within_r1"][m].mean()),
            "chance_r1": float(rd["chance_r1"][m].mean()),
            "mean_cheb": float(rd["cheb"][m].mean()),
            "exp_d_cheb": float(rd["exp_d_cheb"][m].mean()),
            "uniform_null_exp_d": float(rd["uniform_null"][m].mean()),
        }
    return out


def aggregate_buckets(per_seed_breakdowns: list) -> dict:
    """mean/std/min/max across seeds, per bucket, per metric."""
    metrics = ["exact_acc", "within_r1_acc", "chance_r1", "mean_cheb", "exp_d_cheb", "uniform_null_exp_d"]
    agg = {}
    for name in BUCKET_NAMES:
        rows = [bd[name] for bd in per_seed_breakdowns if bd[name] is not None]
        entry = {"n_per_seed": [bd[name]["n"] for bd in per_seed_breakdowns if bd[name] is not None]}
        for met in metrics:
            vals = np.array([r[met] for r in rows], dtype=np.float64)
            entry[met + "_mean"] = float(vals.mean())
            entry[met + "_std"] = float(vals.std())
            entry[met + "_min"] = float(vals.min())
            entry[met + "_max"] = float(vals.max())
            entry[met + "_per_seed"] = vals.tolist()
        agg[name] = entry
    return agg


# ---------------------------------------------------------------------------
# Synthetic "blurred one-hot" reference
# ---------------------------------------------------------------------------
BLUR_SIGMAS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 9.0]
BLUR_SIGMAS_HEADLINE = [1.0, 2.0, 3.0]  # the three the spec asked to report as the primary table


def build_blur_probs(label: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur (Euclidean grid distance) centered on the true cell, clipped to
    the 9x9 grid (no wraparound -- edge/corner cells lose mass to off-grid neighbors,
    same boundary effect as the real per-row chance calculation) and renormalized."""
    w = np.exp(-0.5 * (DIST_EUC[label] / sigma) ** 2)  # [n, 81]
    w = w / w.sum(axis=1, keepdims=True)
    return w.astype(np.float64)


def analyze_blur(label: np.ndarray, bucket: np.ndarray, sigma: float) -> dict:
    probs = build_blur_probs(label, sigma)
    pred = probs.argmax(axis=1)
    cheb = DIST_CHEB[pred, label]
    exact_acc = float((pred == label).mean())
    within_r1_acc = float((cheb <= 1).mean())
    mean_cheb = float(cheb.mean())
    dist_to_label_cheb = DIST_CHEB[:, label]
    exp_d_cheb = float(np.einsum("nc,cn->n", probs, dist_to_label_cheb).mean())

    per_bucket = {}
    for b, name in enumerate(BUCKET_NAMES):
        m = bucket == b
        if not m.any():
            per_bucket[name] = None
            continue
        exp_d_b = float(np.einsum("nc,cn->n", probs[m], dist_to_label_cheb[:, m]).mean())
        per_bucket[name] = {
            "n": int(m.sum()),
            "exact_acc": float((pred[m] == label[m]).mean()),
            "within_r1_acc": float((cheb[m] <= 1).mean()),
            "mean_cheb": float(cheb[m].mean()),
            "exp_d_cheb": exp_d_b,
        }
    return {
        "sigma_blur": sigma,
        "exact_acc": exact_acc,
        "within_r1_acc": within_r1_acc,
        "mean_cheb": mean_cheb,
        "exp_d_cheb": exp_d_cheb,
        "per_bucket": per_bucket,
    }


def main():
    results = {}
    graded_h2 = {}
    per_spec_agg = {}

    for spec_name, files in SPECS.items():
        per_seed_bd = []
        for path, seed in files:
            rd = per_row_data(path)
            bd = bucket_breakdown(rd)
            per_seed_bd.append(bd)
        agg = aggregate_buckets(per_seed_bd)
        per_spec_agg[spec_name] = agg

        # Study 1's own h2_curve() shape test, applied to (a) exact acc [reproduction]
        # and (b) within-r1 acc [exploratory graded recomputation].
        exact_by_bucket = {name: agg[name]["exact_acc_mean"] for name in BUCKET_NAMES}
        r1_by_bucket = {name: agg[name]["within_r1_acc_mean"] for name in BUCKET_NAMES}
        graded_h2[spec_name] = {
            "exact_acc_h2": h2_curve(exact_by_bucket),
            "within_r1_acc_h2_EXPLORATORY": h2_curve(r1_by_bucket),
        }
        results[spec_name] = agg

    # --- correctness check against RESULTS.md H2 table ---
    bdh_full_exact = [results["BDH sigma_full (best L/seed)"][n]["exact_acc_mean"] for n in BUCKET_NAMES]
    rwkv_exact = [results["RWKV state_vec"][n]["exact_acc_mean"] for n in BUCKET_NAMES]
    results_md_bdh_g100 = [0.097, 0.116, 0.105, 0.084, 0.082, 0.101]
    results_md_rwkv = [0.322, 0.271, 0.184, 0.080, 0.029, 0.010]
    bdh_match = np.allclose(bdh_full_exact, results_md_bdh_g100, atol=0.001)
    rwkv_match = np.allclose(rwkv_exact, results_md_rwkv, atol=0.001)

    # --- synthetic blurred one-hot reference (using BDH sigma_full seed0 labels/buckets,
    #     which are shared identically across every spec and seed per the prior report) ---
    ref = np.load(SPECS["BDH sigma_full (best L/seed)"][0][0])
    ref_label = ref["label"].astype(np.int64)
    ref_bucket = ref["bucket"].astype(np.int64)
    blur_results = {str(s): analyze_blur(ref_label, ref_bucket, s) for s in BLUR_SIGMAS}

    # Note: for a symmetric Gaussian blur centered exactly on the true cell, argmax(probs)
    # always recovers the true cell exactly (density is strictly maximal at distance 0), so
    # exact_acc == 1.0 and within_r1_acc == 1.0 and mean_cheb == 0.0 for EVERY sigma_blur --
    # those three are degenerate/uninformative for calibration under this construction. Only
    # E[d] (expected Chebyshev distance under the FULL blurred distribution, not just its
    # argmax) varies with sigma_blur, so that is the only axis usable to match a real spec's
    # coarseness in cells.
    blur_grid_sigma = np.array(BLUR_SIGMAS)
    blur_grid_expd = np.array([blur_results[str(s)]["exp_d_cheb"] for s in BLUR_SIGMAS])

    def nearest_blur_sigma(exp_d_value, uniform_null_value=None):
        """Linear-interpolate the sigma_blur (in cells) whose synthetic E[d] equals exp_d_value.
        A Gaussian blur's E[d] increases monotonically toward (but never reaches or exceeds) the
        uniform-null distance as sigma_blur -> infinity, so a value at or above that row's own
        uniform null cannot be matched by ANY blur scale -- flagged explicitly rather than clipped
        to the sampled grid's max, since that would misrepresent an unmatchable value as sigma=9."""
        if uniform_null_value is not None and exp_d_value >= uniform_null_value:
            return None, "at or above uniform null -- no blur scale matches (worse than random)"
        if exp_d_value <= blur_grid_expd.min():
            return float(blur_grid_sigma[np.argmin(blur_grid_expd)]), "below grid (clipped)"
        if exp_d_value >= blur_grid_expd.max():
            return float(blur_grid_sigma[np.argmax(blur_grid_expd)]), "above grid (clipped)"
        return float(np.interp(exp_d_value, blur_grid_expd, blur_grid_sigma)), "interpolated"

    # Overall (non-bucketed) E[d] values, recomputed here from the per-bucket breakdown as an
    # n-weighted average so they are internally consistent with this script's own numbers.
    def overall_expd(spec_name):
        agg = results[spec_name]
        ns = np.array([agg[n]["n_per_seed"][0] for n in BUCKET_NAMES], dtype=np.float64)
        vs = np.array([agg[n]["exp_d_cheb_mean"] for n in BUCKET_NAMES], dtype=np.float64)
        return float((ns * vs).sum() / ns.sum())

    def overall_null(spec_name):
        agg = results[spec_name]
        ns = np.array([agg[n]["n_per_seed"][0] for n in BUCKET_NAMES], dtype=np.float64)
        vs = np.array([agg[n]["uniform_null_exp_d_mean"] for n in BUCKET_NAMES], dtype=np.float64)
        return float((ns * vs).sum() / ns.sum())

    calibration = {}
    for spec_name in ["BDH sigma_full (best L/seed)", "BDH sigma_rownorm (best L/seed)",
                       "LSTM state_vec", "RWKV state_vec"]:
        overall_e = overall_expd(spec_name)
        overall_n = overall_null(spec_name)
        sig, note = nearest_blur_sigma(overall_e, overall_n)
        per_bucket_cal = {}
        for name in BUCKET_NAMES:
            e = results[spec_name][name]["exp_d_cheb_mean"]
            nul = results[spec_name][name]["uniform_null_exp_d_mean"]
            s, n = nearest_blur_sigma(e, nul)
            per_bucket_cal[name] = {"exp_d_cheb": e, "uniform_null": nul, "nearest_blur_sigma": s, "note": n}
        calibration[spec_name] = {
            "overall_exp_d_cheb": overall_e, "overall_uniform_null": overall_n,
            "nearest_blur_sigma_overall": sig, "note": note,
            "per_bucket": per_bucket_cal,
        }

    out = {
        "bucket_names": BUCKET_NAMES,
        "per_spec_bucket_aggregate": results,
        "graded_h2_shape_test": graded_h2,
        "correctness_check": {
            "bdh_sigma_full_exact_by_bucket": bdh_full_exact,
            "results_md_bdh_g100": results_md_bdh_g100,
            "bdh_match": bool(bdh_match),
            "rwkv_state_vec_exact_by_bucket": rwkv_exact,
            "results_md_rwkv": results_md_rwkv,
            "rwkv_match": bool(rwkv_match),
        },
        "blurred_one_hot_reference": blur_results,
        "blur_calibration": calibration,
    }
    (OUT_DIR / "spatial_locality_buckets_results.json").write_text(json.dumps(out, indent=2))
    print("Wrote", OUT_DIR / "spatial_locality_buckets_results.json")

    print(f"\nCorrectness check: BDH sigma_full exact-by-bucket matches RESULTS.md bdh_g100 row: {bdh_match}")
    print(f"  computed: {[round(v, 3) for v in bdh_full_exact]}")
    print(f"  RESULTS.md: {results_md_bdh_g100}")
    print(f"Correctness check: RWKV state_vec exact-by-bucket matches RESULTS.md rwkv row: {rwkv_match}")
    print(f"  computed: {[round(v, 3) for v in rwkv_exact]}")
    print(f"  RESULTS.md: {results_md_rwkv}")

    for spec_name, agg in results.items():
        print(f"\n=== {spec_name} ===")
        for name in BUCKET_NAMES:
            e = agg[name]
            print(f"  {name:>6} (n={e['n_per_seed']}): exact={e['exact_acc_mean']:.3f}±{e['exact_acc_std']:.3f}"
                  f"  within_r1={e['within_r1_acc_mean']:.3f}±{e['within_r1_acc_std']:.3f}"
                  f" (chance={e['chance_r1_mean']:.3f})"
                  f"  mean_cheb={e['mean_cheb_mean']:.3f}±{e['mean_cheb_std']:.3f}"
                  f"  E[d]={e['exp_d_cheb_mean']:.3f} (uniform_null={e['uniform_null_exp_d_mean']:.3f})")
        g_exact = graded_h2[spec_name]["exact_acc_h2"]
        g_r1 = graded_h2[spec_name]["within_r1_acc_h2_EXPLORATORY"]
        print(f"  H2 shape test (exact acc, reproduction): graceful={g_exact['graceful']} "
              f"ratio(33-64/1-4)={g_exact['ratio_33_64_over_1_4']}")
        print(f"  H2 shape test (within-r1 acc, EXPLORATORY): graceful={g_r1['graceful']} "
              f"ratio(33-64/1-4)={g_r1['ratio_33_64_over_1_4']}")

    print("\n=== Blurred one-hot reference (Gaussian blur of true cell, Euclidean, clipped to 9x9 grid) ===")
    for s in BLUR_SIGMAS:
        r = blur_results[str(s)]
        print(f"  sigma_blur={s}: exact_acc={r['exact_acc']:.3f}  within_r1_acc={r['within_r1_acc']:.3f}"
              f"  mean_cheb={r['mean_cheb']:.3f}  E[d]={r['exp_d_cheb']:.3f}"
              f"{'  <- headline' if s in BLUR_SIGMAS_HEADLINE else ''}")
    print("  NOTE: exact_acc/within_r1_acc/mean_cheb are trivially 1.0/1.0/0.0 for every sigma_blur")
    print("  because a symmetric blur's argmax always lands exactly on the true cell. Only E[d]")
    print("  (uses the full distribution, not argmax) is sensitive to blur scale.")

    def _fmt_sigma(s):
        return f"{s:.2f}" if s is not None else "n/a"

    print("\n=== Blur-scale calibration (nearest sigma_blur, in cells, matched via E[d]) ===")
    for spec_name, cal in calibration.items():
        print(f"  {spec_name}: overall E[d]={cal['overall_exp_d_cheb']:.3f} "
              f"-> nearest sigma_blur={_fmt_sigma(cal['nearest_blur_sigma_overall'])} cells ({cal['note']})")
        for name in BUCKET_NAMES:
            pb = cal["per_bucket"][name]
            print(f"      {name:>6}: E[d]={pb['exp_d_cheb']:.3f} (null={pb['uniform_null']:.3f}) "
                  f"-> sigma_blur={_fmt_sigma(pb['nearest_blur_sigma'])} ({pb['note']})")


if __name__ == "__main__":
    main()
