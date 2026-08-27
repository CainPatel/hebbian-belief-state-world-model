"""
Study 1 post-hoc exploratory analysis: is the sigma memory's probe error
spatially LOCAL (coarse spatial code) rather than scattered (no code at all)?

Reads only already-saved probe .npz outputs under runs/study1/ and the
probe_test split under data/grid9/. No model inference. Post-hoc / descriptive,
not a preregistered claim.
"""
import json
import os
from pathlib import Path

import numpy as np

# ROOT must contain runs/ and data/. Study 1's artifacts live in the sibling worktree, which is the
# default below; override with HBWM_ROOT to point at any other checkout of the same layout.
# OUT_DIR receives the JSON dump; defaults to the current working directory.
_REPO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("HBWM_ROOT", _REPO / ".claude/worktrees/study1-impl"))
OUT_DIR = Path(os.environ.get("HBWM_POSTHOC_OUT", Path.cwd()))

G = 9
N_CLASSES = G * G
SEEDS = [0, 1, 2]
RNG_SEED = 0
N_SHUFFLES = 20

# ---- per-seed spec selection (from runs/study1/results/table.json, bdh_g100) ----
# sigma_full levels : [3, 3, 4]   (matches probes/done.json best_full_spec per seed)
# sigma_rownorm levels: [4, 3, 4]  (best test_acc level per seed, per table.json)
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

PROBE_TEST_NPZ = ROOT / "data/grid9/probe_test.npz"

# ---------------------------------------------------------------------------
# Distance machinery over the 81-cell grid (cell = y*G + x)
# ---------------------------------------------------------------------------
_c = np.arange(N_CLASSES)
_X = _c % G
_Y = _c // G
DX = np.abs(_X[:, None] - _X[None, :])
DY = np.abs(_Y[:, None] - _Y[None, :])
DIST_CHEB = np.maximum(DX, DY).astype(np.float64)   # [81,81]
DIST_MANH = (DX + DY).astype(np.float64)             # [81,81]

NEIGH_R1_COUNT = (DIST_CHEB <= 1).sum(axis=1)  # per true-cell count of cells within radius 1 (incl. self)
NEIGH_R2_COUNT = (DIST_CHEB <= 2).sum(axis=1)
ROW_MEAN_DIST_CHEB = DIST_CHEB.mean(axis=1)    # E[dist] under a uniform predictive dist, per true cell
ROW_MEAN_DIST_MANH = DIST_MANH.mean(axis=1)


def cell_xy(c):
    return c % G, c // G


def load_agent_cells():
    """agent_cell[ep, t] = cell id of the agent's own position at that step."""
    d = np.load(PROBE_TEST_NPZ)
    ap = d["agent_pos"]  # [n_ep, T, 2] as (x, y) -- see hbwm/instrument/atlas.py convention
    agent_cell = ap[..., 1] * G + ap[..., 0]
    return agent_cell.astype(np.int64)


AGENT_CELL = load_agent_cells()  # [2000, 97]


def analyze_spec(npz_path: Path) -> dict:
    d = np.load(npz_path)
    probs = d["probs"].astype(np.float32)  # [n, 81]
    label = d["label"].astype(np.int64)
    ep = d["ep"].astype(np.int64)
    t = d["t"].astype(np.int64)
    n = len(label)

    pred = probs.argmax(axis=1)

    cheb = DIST_CHEB[pred, label]
    manh = DIST_MANH[pred, label]

    # --- 1. error locality ---
    dist_hist = {
        "cheb_0": float((cheb == 0).mean()),
        "cheb_1": float((cheb == 1).mean()),
        "cheb_2": float((cheb == 2).mean()),
        "cheb_3plus": float((cheb >= 3).mean()),
    }
    mean_cheb = float(cheb.mean())
    mean_manh = float(manh.mean())

    # --- 2. within-radius accuracy vs. its own chance rate ---
    within_r1 = cheb <= 1
    within_r2 = cheb <= 2
    within_r1_acc = float(within_r1.mean())
    within_r2_acc = float(within_r2.mean())
    chance_r1 = float((NEIGH_R1_COUNT[label] / N_CLASSES).mean())
    chance_r2 = float((NEIGH_R2_COUNT[label] / N_CLASSES).mean())

    # --- 3. expected distance under the full predictive distribution ---
    # E[d]_i = sum_c probs[i,c] * dist(c, label_i)
    dist_to_label_cheb = DIST_CHEB[:, label]  # [81, n]
    dist_to_label_manh = DIST_MANH[:, label]  # [81, n]
    exp_d_cheb = np.einsum("nc,cn->n", probs, dist_to_label_cheb)
    exp_d_manh = np.einsum("nc,cn->n", probs, dist_to_label_manh)
    mean_exp_d_cheb = float(exp_d_cheb.mean())
    mean_exp_d_manh = float(exp_d_manh.mean())

    uniform_exp_d_cheb = float(ROW_MEAN_DIST_CHEB[label].mean())
    uniform_exp_d_manh = float(ROW_MEAN_DIST_MANH[label].mean())

    rng = np.random.default_rng(RNG_SEED)
    shuf_means_cheb = []
    shuf_means_manh = []
    for _ in range(N_SHUFFLES):
        perm = rng.permutation(n)
        probs_shuf = probs[perm]
        e_cheb = np.einsum("nc,cn->n", probs_shuf, dist_to_label_cheb).mean()
        e_manh = np.einsum("nc,cn->n", probs_shuf, dist_to_label_manh).mean()
        shuf_means_cheb.append(e_cheb)
        shuf_means_manh.append(e_manh)
    shuf_mean_cheb = float(np.mean(shuf_means_cheb))
    shuf_std_cheb = float(np.std(shuf_means_cheb))
    shuf_mean_manh = float(np.mean(shuf_means_manh))
    shuf_std_manh = float(np.std(shuf_means_manh))

    # --- 4. marginal structure (x, y decoded separately) ---
    pred_x, pred_y = cell_xy(pred)
    true_x, true_y = cell_xy(label)
    x_acc = float((pred_x == true_x).mean())
    y_acc = float((pred_y == true_y).mean())
    x_mae = float(np.abs(pred_x - true_x).mean())
    y_mae = float(np.abs(pred_y - true_y).mean())

    # --- 5. top-k ---
    order = np.argsort(-probs, axis=1)
    top3 = order[:, :3]
    top5 = order[:, :5]
    top3_acc = float((top3 == label[:, None]).any(axis=1).mean())
    top5_acc = float((top5 == label[:, None]).any(axis=1).mean())

    # --- 6. agent-proximity control ---
    agent_cell = AGENT_CELL[ep, t]
    pred_to_agent_cheb = DIST_CHEB[pred, agent_cell]
    label_to_agent_cheb = DIST_CHEB[label, agent_cell]
    mean_pred_agent_cheb = float(pred_to_agent_cheb.mean())
    mean_label_agent_cheb = float(label_to_agent_cheb.mean())
    frac_pred_is_agent_cell = float((pred == agent_cell).mean())
    frac_label_is_agent_cell = float((label == agent_cell).mean())
    exact_acc = float((pred == label).mean())

    return {
        "n": int(n),
        "exact_acc": exact_acc,
        "mean_cheb": mean_cheb,
        "mean_manh": mean_manh,
        **dist_hist,
        "within_r1_acc": within_r1_acc,
        "chance_r1": chance_r1,
        "within_r2_acc": within_r2_acc,
        "chance_r2": chance_r2,
        "exp_d_cheb": mean_exp_d_cheb,
        "exp_d_cheb_uniform_null": uniform_exp_d_cheb,
        "exp_d_cheb_shuffle_null": shuf_mean_cheb,
        "exp_d_cheb_shuffle_within_seed_std": shuf_std_cheb,
        "exp_d_manh": mean_exp_d_manh,
        "exp_d_manh_uniform_null": uniform_exp_d_manh,
        "exp_d_manh_shuffle_null": shuf_mean_manh,
        "exp_d_manh_shuffle_within_seed_std": shuf_std_manh,
        "x_acc": x_acc,
        "y_acc": y_acc,
        "x_mae": x_mae,
        "y_mae": y_mae,
        "top3_acc": top3_acc,
        "top5_acc": top5_acc,
        "mean_pred_to_agent_cheb": mean_pred_agent_cheb,
        "mean_label_to_agent_cheb": mean_label_agent_cheb,
        "frac_pred_is_agent_cell": frac_pred_is_agent_cell,
        "frac_label_is_agent_cell": frac_label_is_agent_cell,
    }


def aggregate(per_seed_results: list) -> dict:
    skip = {"n", "seed", "file"}
    keys = [k for k in per_seed_results[0].keys() if k not in skip]
    agg = {"n_per_seed": [r["n"] for r in per_seed_results]}
    for k in keys:
        vals = np.array([r[k] for r in per_seed_results], dtype=np.float64)
        agg[k + "_mean"] = float(vals.mean())
        agg[k + "_std"] = float(vals.std())
    return agg


def main():
    results = {}
    per_spec_per_seed = {}
    for spec_name, files in SPECS.items():
        seed_results = []
        for path, seed in files:
            r = analyze_spec(path)
            r["seed"] = seed
            r["file"] = str(path.relative_to(ROOT))
            seed_results.append(r)
        per_spec_per_seed[spec_name] = seed_results
        results[spec_name] = aggregate(seed_results)

    out = {
        "per_seed": per_spec_per_seed,
        "aggregate": results,
        "chance_top3": 3.0 / N_CLASSES,
        "chance_top5": 5.0 / N_CLASSES,
        "chance_exact": 1.0 / N_CLASSES,
        "n_shuffles_for_null": N_SHUFFLES,
    }
    (OUT_DIR / "spatial_locality_results.json").write_text(json.dumps(out, indent=2))
    print("Wrote", OUT_DIR / "spatial_locality_results.json")

    # quick console summary
    for spec_name, agg in results.items():
        print(f"\n=== {spec_name} ===")
        print(f"  n/seed: {agg['n_per_seed']}")
        print(f"  exact acc: {agg['exact_acc_mean']:.3f} ± {agg['exact_acc_std']:.3f}")
        print(f"  mean Chebyshev err: {agg['mean_cheb_mean']:.3f} ± {agg['mean_cheb_std']:.3f}")
        print(f"  within-r1 acc: {agg['within_r1_acc_mean']:.3f} ± {agg['within_r1_acc_std']:.3f}"
              f"  (chance {agg['chance_r1_mean']:.3f})")
        print(f"  within-r2 acc: {agg['within_r2_acc_mean']:.3f} ± {agg['within_r2_acc_std']:.3f}"
              f"  (chance {agg['chance_r2_mean']:.3f})")
        print(f"  E[d] full: {agg['exp_d_cheb_mean']:.3f}  uniform-null: {agg['exp_d_cheb_uniform_null_mean']:.3f}"
              f"  shuffle-null: {agg['exp_d_cheb_shuffle_null_mean']:.3f}")
        print(f"  frac pred==agent_cell: {agg['frac_pred_is_agent_cell_mean']:.3f}   "
              f"frac label==agent_cell: {agg['frac_label_is_agent_cell_mean']:.3f}")
        print(f"  pred->agent dist: {agg['mean_pred_to_agent_cheb_mean']:.3f}   "
              f"label->agent dist: {agg['mean_label_to_agent_cheb_mean']:.3f}")


if __name__ == "__main__":
    main()
