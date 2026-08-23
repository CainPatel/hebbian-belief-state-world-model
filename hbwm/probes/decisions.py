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
    frac = (sum(1 for l in lat if l <= 5) / n_ep) if n_ep else float("nan")
    return {"n_episodes": int(n_ep), "n_flipped": len(lat), "latencies": lat,
            "median_latency": (float(np.median(lat)) if lat else None), "frac_le5": frac, "supported": bool(frac >= 0.7)}


def h4_k90(acc_by_k, acc_all, n_features):
    ks = sorted(int(k) for k in acc_by_k)
    k90 = next((k for k in ks if acc_by_k[k] >= 0.9 * acc_all), None)
    return {"k90": k90, "k90_frac": (k90 / n_features if k90 else None),
            "strong": bool(k90 is not None and k90 <= 256), "weak": bool(k90 is not None and k90 <= 0.01 * n_features)}
