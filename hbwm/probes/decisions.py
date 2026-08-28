"""Preregistered decision rules (spec section 4.5). Pure functions over numbers; no I/O."""

import numpy as np

from hbwm.probes.eligibility import BUCKET_NAMES


def h1_decision(bdh_acc, comparators, margin=0.05):
    bdh = np.asarray(bdh_acc, dtype=float)
    out = {"supported": True, "margin": margin, "bdh_mean": float(bdh.mean()), "comparators": {}}
    for name, accs in comparators.items():
        c = np.asarray(accs, dtype=float)
        diffs = bdh - c
        passes = bool(bdh.mean() - c.mean() > margin and (diffs > 0).all())
        out["comparators"][name] = {"mean": float(c.mean()), "mean_diff": float(bdh.mean() - c.mean()),
                                    "paired_diffs": diffs.tolist(), "passes": passes}
        out["supported"] = out["supported"] and passes
    return out


def h2_curve(bucket_acc):
    vals = [(n, bucket_acc.get(n)) for n in BUCKET_NAMES]
    present = [(n, v) for n, v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    acc = dict(present)
    a14, a3364 = acc.get("1-4"), acc.get("33-64")
    ratio = (a3364 / a14) if (a14 and a3364 is not None) else None
    no_cliff = all(b >= 0.5 * a for (_, a), (_, b) in zip(present, present[1:]))
    graceful = bool(ratio is not None and ratio >= 0.5 and no_cliff)
    return {"values": {n: v for n, v in present}, "graceful": graceful, "ratio_33_64_over_1_4": ratio}


def h3_latency(p_old, p_new, steps, ep):
    p_old, p_new, steps, ep = map(np.asarray, (p_old, p_new, steps, ep))
    lat = []
    n_ep = 0
    for e in np.unique(ep):
        m = ep == e
        n_ep += 1
        order = np.argsort(steps[m])
        flips = np.where((p_new[m] > p_old[m])[order])[0]
        if len(flips):
            lat.append(int(steps[m][order][flips[0]]))
    frac = (sum(1 for v in lat if v <= 5) / n_ep) if n_ep else float("nan")
    return {"n_episodes": int(n_ep), "n_flipped": len(lat), "latencies": lat,
            "median_latency": (float(np.median(lat)) if lat else None), "frac_le5": frac, "supported": bool(frac >= 0.7)}


def h4_k90(acc_by_k, acc_all, n_features):
    """k90 = min k on the grid with acc ≥ 0.9 · acc(all). The spec 4.5 grid ends in "all", i.e. k =
    n_features with accuracy acc_all, so that terminal point always qualifies: when no listed k reaches
    the threshold, k90 = n_features and k90_frac = 1.0 (never None), which is neither strong nor weak
    for any realistic feature count. `acc_by_k` keys may be ints or JSON string keys."""
    acc = {int(k): v for k, v in acc_by_k.items()}
    k90 = next((k for k in sorted(acc) if acc[k] >= 0.9 * acc_all), n_features)
    return {"k90": k90, "k90_frac": k90 / n_features,
            "strong": bool(k90 <= 256), "weak": bool(k90 <= 0.01 * n_features)}


def _paired(a, b, margin):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError(f"_paired: length mismatch, got {len(a)} and {len(b)}")
    diffs = a - b
    return {"mean": float(b.mean()), "mean_diff": float(a.mean() - b.mean()),
            "paired_diffs": diffs.tolist(),
            "passes": bool(a.mean() - b.mean() > margin and (diffs > 0).all())}


def h5_decision(structured_accs, flat_accs, margin=0.05):
    """Spec 7 H5: best structured sigma readout (families 2 to 4) vs flat_linear on sigma."""
    r = _paired(structured_accs, flat_accs, margin)
    return {"supported": r["passes"], "margin": margin,
            "structured_mean": float(np.mean(structured_accs)), "flat_mean": r["mean"],
            "mean_diff": r["mean_diff"], "paired_diffs": r["paired_diffs"]}


def h6_decision(bdh_accs, baselines, family, saturated=None, margin=0.05):
    """Spec 7 H6: the headline, within the best matched family.

    `saturated` maps a baseline name to whether its arm of this family is rank-saturated (spec 5.2). A
    win over a saturated arm is flagged as a possible rank-constraint artifact; the flag does not change
    `supported`, it changes how the verdict must be read.
    """
    saturated = saturated or {}
    missing = sorted({"lstm", "rwkv"} - baselines.keys())
    if missing:
        raise ValueError(f"h6_decision requires both lstm and rwkv baselines, missing: {missing}")
    out = {"supported": True, "margin": margin, "family": family,
           "bdh_mean": float(np.mean(bdh_accs)), "comparators": {}}
    for name, accs in baselines.items():
        out["comparators"][name] = _paired(bdh_accs, accs, margin)
        out["supported"] = out["supported"] and out["comparators"][name]["passes"]
    out["kill_criterion_fired"] = not out["comparators"]["lstm"]["passes"]
    out["saturated_baselines"] = sorted(n for n, s in saturated.items() if s)
    out["artifact_warning"] = bool(out["supported"] and out["saturated_baselines"])
    return out


def is_degenerate(train_acc, val_acc, chance, train_bar=0.95, val_factor=2.0):
    """Preregistered degeneracy criterion (spec 7). Applies to EVERY arm of every family and model.

    `train_acc` and `val_acc` map "<l2>/<restart>" to accuracy, as the probe JSON records them. An arm
    is degenerate iff, at EVERY l2 value in the grid, the restart that probe_val would select has
    training accuracy above `train_bar` AND validation accuracy below `val_factor` times the
    majority-class chance rate. A degenerate arm is excluded from H6's best-matched-family selection;
    it is still fitted and still reported, with its parameter count, training and validation accuracy,
    and this flag. The point is to stop a probe that has merely memorized its training set from
    deciding a hypothesis.
    """
    out = {"degenerate": False, "per_l2": {}, "train_bar": train_bar,
           "val_bar": val_factor * chance, "chance": chance}
    if not val_acc:
        return out
    best = {}
    for key, v in val_acc.items():
        l2 = key.split("/")[0]
        if l2 not in best or v > val_acc[best[l2]]:
            best[l2] = key
    out["per_l2"] = {
        l2: {"key": k, "train_acc": train_acc[k], "val_acc": val_acc[k],
             "memorizing": bool(train_acc[k] > train_bar and val_acc[k] < out["val_bar"])}
        for l2, k in best.items()
    }
    out["degenerate"] = bool(all(d["memorizing"] for d in out["per_l2"].values()))
    return out


def h7_attribution(mlp_accs, structured_accs, tol=0.02):
    """Spec 7 H7: reported, never gated. Within `tol` or better means capacity, not structure."""
    m, s = float(np.mean(mlp_accs)), float(np.mean(structured_accs))
    return {"mlp_mean": m, "structured_mean": s, "diff": m - s, "tol": tol,
            "attribute_to_capacity": bool(m >= s - tol), "gates_nothing": True}


def h8_latency(p_old, p_new, steps, ep, visible_now, exclusion_cap=0.25):
    """Spec 7 H8: latency from t0, the first not-visible step at or after re-observation.

    Episodes with no such t0 (the object never leaves the 3x3 window before the episode ends) are
    EXCLUDED from the denominator, not counted as failures: no belief-revision test is possible there.
    Episodes that have a t0 but never flip ARE failures. Above `exclusion_cap` the result is flagged
    low-coverage.
    """
    p_old, p_new, steps, ep, vis = map(np.asarray, (p_old, p_new, steps, ep, visible_now))
    vis = vis.astype(bool)
    lat, n_scored, n_excluded = [], 0, 0
    for e in np.unique(ep):
        m = ep == e
        order = np.argsort(steps[m])
        s_e, v_e = steps[m][order], vis[m][order]
        free = np.where(~v_e)[0]
        if len(free) == 0:
            n_excluded += 1
            continue
        n_scored += 1
        t0_pos, t0 = free[0], s_e[free[0]]
        flips = np.where((p_new[m][order] > p_old[m][order])[t0_pos:])[0]
        if len(flips):
            lat.append(int(s_e[t0_pos + flips[0]] - t0))
    total = n_scored + n_excluded
    frac = (sum(1 for v in lat if v <= 5) / n_scored) if n_scored else float("nan")
    excluded_frac = (n_excluded / total) if total else 0.0
    return {"n_episodes": n_scored, "n_excluded": n_excluded, "n_total": total,
            "excluded_frac": excluded_frac, "low_coverage": bool(excluded_frac > exclusion_cap),
            "n_flipped": len(lat), "latencies": lat,
            "median_latency": (float(np.median(lat)) if lat else None),
            "frac_le5": frac, "supported": bool(n_scored and frac >= 0.7)}
