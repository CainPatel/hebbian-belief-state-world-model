"""Cross-seed aggregation of probe results; applies the preregistered rules; writes results/."""

import argparse
import json
from pathlib import Path

import numpy as np

from hbwm.matrix import GAMMA_ARMS, best_lr, run_path
from hbwm.probes.decisions import h1_decision, h2_curve, h3_latency, h4_k90
from hbwm.probes.eligibility import BUCKET_NAMES
from hbwm.probes.run import val_best

BDH_STEMS = ["bdh_g100"] + GAMMA_ARMS
BASELINES = ["lstm", "rwkv"]


def load_run(run_dir) -> dict:
    run_dir = Path(run_dir)
    pdir = run_dir / "probes"
    probes = {p.stem: json.loads(p.read_text()) for p in pdir.glob("*.json") if p.name != "done.json"}
    h3 = {p.name[: -len("_h3.npz")]: dict(np.load(p)) for p in pdir.glob("*_h3.npz")}
    return {"final": json.loads((run_dir / "final.json").read_text()),
            "done": json.loads((pdir / "done.json").read_text()) if (pdir / "done.json").exists() else {},
            "probes": probes, "h3": h3}


def best_level(probes: dict, feature: str):
    cands = {k: r for k, r in probes.items() if k == feature or k.startswith(feature + "_L")}
    if not cands:
        return None, None
    # highest val accuracy; ties broken by lowest level, matching run.py's select_best_full
    k = min(cands, key=lambda n: (-val_best(cands[n]), cands[n]["level"] if cands[n]["level"] is not None else -1))
    return k, cands[k]


def _seed_runs(root, exp, stem):
    lr = best_lr(root, exp, "bdh_g100" if stem in GAMMA_ARMS else stem)
    return {seed: load_run(run_path(root, exp, stem, lr, seed)) for seed in (0, 1, 2)}


def aggregate(root, exp, data_dir=None) -> dict:
    runs = {stem: _seed_runs(root, exp, stem) for stem in BDH_STEMS + BASELINES}
    table = {}
    for stem, by_seed in runs.items():
        feats = ["sigma_full", "sigma_rownorm", "x_sparse", "resid"] if stem in BDH_STEMS else ["state_vec"]
        table[stem] = {}
        for f in feats:
            accs, levels, specs, cis = [], [], [], []
            for seed in sorted(by_seed):
                name, r = best_level(by_seed[seed]["probes"], f)
                if r is None:
                    continue
                accs.append(r["test_acc"])
                levels.append(r["level"])
                specs.append(name)
                cis.append(r["ci95"])
            if accs:
                table[stem][f] = {"per_seed": accs, "mean": float(np.mean(accs)), "std": float(np.std(accs)),
                                  "levels": levels, "specs": specs, "ci95": cis,
                                  "n_train": by_seed[0]["probes"][specs[0]].get("n_train"),
                                  "n_features": by_seed[0]["probes"][specs[0]]["n_features"],
                                  "chance": by_seed[0]["probes"][specs[0]]["chance"],
                                  "ceiling": by_seed[0]["probes"][specs[0]]["ceiling"]}
    # H1
    h1 = h1_decision(table["bdh_g100"]["sigma_full"]["per_seed"],
                     {"x_sparse": table["bdh_g100"]["x_sparse"]["per_seed"],
                      "lstm": table["lstm"]["state_vec"]["per_seed"], "rwkv": table["rwkv"]["state_vec"]["per_seed"]})
    # H2: mean bucket accuracy over seeds of the chosen spec
    h2 = {}
    for stem in BDH_STEMS + BASELINES:
        f = "sigma_full" if stem in BDH_STEMS else "state_vec"
        per_bucket = {b: [] for b in BUCKET_NAMES}
        for seed, spec in zip(sorted(runs[stem]), table[stem][f]["specs"]):
            for b, v in runs[stem][seed]["probes"][spec]["bucket_acc"].items():
                if v is not None:
                    per_bucket[b].append(v)
        mean_b = {b: (float(np.mean(v)) if v else None) for b, v in per_bucket.items()}
        # every seed probes the same probe_test pairs, so seed 0's chosen spec carries the bucket counts
        s0 = min(runs[stem])
        bucket_n = runs[stem][s0]["probes"][table[stem][f]["specs"][0]].get("bucket_n") or {}
        h2[stem] = {**h2_curve(mean_b), "spec_per_seed": table[stem][f]["specs"], "bucket_n": bucket_n}
    # H3 / H4 on each run's selected spec
    h3, h4 = {}, {}
    for stem in BDH_STEMS + BASELINES:
        fr, fr_nv, k90s, per_seed_h3, per_seed_h4 = [], [], [], {}, {}
        for seed, run in runs[stem].items():
            for spec, arrs in run["h3"].items():
                d = h3_latency(arrs["p_old"], arrs["p_new"], arrs["steps_since_reobs"], arrs["ep"])
                per_seed_h3[seed] = {"spec": spec, **{k: v for k, v in d.items() if k != "latencies"}}
                fr.append(d["frac_le5"])
                if "visible_now" in arrs:  # exploratory: same rule, restricted to still-unseen steps
                    nv = ~np.asarray(arrs["visible_now"], dtype=bool)
                    e = h3_latency(arrs["p_old"][nv], arrs["p_new"][nv], arrs["steps_since_reobs"][nv], arrs["ep"][nv])
                    per_seed_h3[seed]["exploratory_not_visible"] = {
                        k: e[k] for k in ("n_episodes", "n_flipped", "median_latency", "frac_le5")}
                    fr_nv.append(e["frac_le5"])
            had_h4 = False
            for spec, r in run["probes"].items():
                if "h4" in r:
                    had_h4 = True
                    d = h4_k90(r["h4"]["acc_by_k"], r["h4"]["acc_all"], r["n_features"])
                    neurons_by_k = r["h4"]["neurons_by_k"] or {}
                    # k90 = n_features is the terminal fallback (spec 4.5 grid ends in "all"; no listed k
                    # reached the threshold), a key never present in neurons_by_k, which only lists the
                    # explicit k grid. Fall back to the run's total neuron count (None for non-BDH baselines).
                    if d["k90"] == r["n_features"] or str(d["k90"]) not in neurons_by_k:
                        d["neurons_at_k90"] = r["h4"]["n_neurons_total"]
                    else:
                        d["neurons_at_k90"] = neurons_by_k[str(d["k90"])]
                    per_seed_h4[seed] = {"spec": spec, "acc_by_k": r["h4"]["acc_by_k"], **d}
                    k90s.append(d["k90"])
            if not had_h4:  # a seed with no H4 at all is a seed whose k90 was never reached
                k90s.append(float("inf"))
        if per_seed_h3:
            m = float(np.mean(fr))
            h3[stem] = {"per_seed": per_seed_h3, "mean_frac_le5": m, "supported": bool(m >= 0.7),
                        "mean_frac_le5_not_visible": (float(np.mean(fr_nv)) if fr_nv else None)}
        if per_seed_h4:
            med = float(np.median(k90s))
            nf = next(iter(per_seed_h4.values()))
            n_features = runs[stem][next(iter(per_seed_h4))]["probes"][nf["spec"]]["n_features"]
            h4[stem] = {"per_seed": per_seed_h4, "median_k90": (None if np.isinf(med) else med),
                        "strong": bool(med <= 256), "weak": bool(med <= 0.01 * n_features), "n_features": n_features}
    perplexity = {}
    for stem, by_seed in runs.items():
        v = [r["final"]["best_val_ce"] for r in by_seed.values()]
        perplexity[stem] = {"val_ce_mean": float(np.mean(v)), "val_ce_std": float(np.std(v)),
                            "n_params": by_seed[0]["final"].get("n_params"), "lr": by_seed[0]["final"].get("lr")}
        if data_dir is not None:
            tc = [test_ce(run_path(root, exp, stem, best_lr(root, exp, "bdh_g100" if stem in GAMMA_ARMS else stem), s), data_dir)
                  for s in by_seed]
            perplexity[stem]["test_ce_mean"] = float(np.mean([t["test_ce"] for t in tc]))
            perplexity[stem]["test_ce_window_mean"] = float(np.mean([t["test_ce_window"] for t in tc]))
    return {"table": table, "h1": h1, "h2": h2, "h3": h3, "h4": h4, "perplexity": perplexity}


def test_ce(run_dir, data_dir, device=None):
    import torch  # noqa: F401

    from hbwm.device import select_device
    from hbwm.envs.dataset import EpisodeData
    from hbwm.train import TrainConfig, evaluate, load_checkpoint

    dev = select_device(device)
    model, cfg, _ = load_checkpoint(Path(run_dir) / "ckpt.pt", dev)
    d = EpisodeData(data_dir, "probe_test")
    out = evaluate(model, d, TrainConfig(batch_size=32, eval_episodes=d.n), dev)
    return {"test_ce": out["val_ce"], "test_ce_window": out["val_ce_window"]}


def _fmt(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def _val(x):
    """Verbatim cell for a value that is not a fixed-precision float (counts, params, lr)."""
    return "-" if x is None else str(x)


def write_outputs(agg: dict, out_dir) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for k in ("table", "h1", "h2", "h3", "h4", "perplexity"):
        (out / f"{k}.json").write_text(json.dumps(agg[k], indent=2, default=float) + "\n")
    lines = ["## Probe accuracy (test, best level per seed; mean ± std over 3 seeds)", "",
             "| model | feature | acc | chance | ceiling | #features | n_train | levels |",
             "|---|---|---|---|---|---|---|---|"]
    for stem, feats in agg["table"].items():
        for f, r in feats.items():
            levels = "-" if all(lv is None for lv in r["levels"]) else r["levels"]
            lines.append(f"| {stem} | {f} | {r['mean']:.3f} ± {r['std']:.3f} | {r['chance']:.3f} | {r['ceiling']:.3f} "
                         f"| {r['n_features']} | {_val(r.get('n_train'))} | {levels} |")
    h1 = agg["h1"]
    lines += ["", f"## H1 — supported: **{h1['supported']}** (margin {h1['margin']:.2f})", "",
              "| comparator | mean diff | paired diffs | passes |", "|---|---|---|---|"]
    for n, c in h1["comparators"].items():
        lines.append(f"| {n} | {c['mean_diff']:+.3f} | {[round(d, 3) for d in c['paired_diffs']]} | {c['passes']} |")
    lines += ["", "## H2 — decay curves (accuracy by steps-since-seen bucket)", "",
              "| model | " + " | ".join(BUCKET_NAMES) + " | graceful |", "|---|" + "---|" * (len(BUCKET_NAMES) + 1)]
    for stem, r in agg["h2"].items():
        lines.append(f"| {stem} | " + " | ".join(_fmt(r["values"].get(b)) for b in BUCKET_NAMES) + f" | {r['graceful']} |")
    lines += ["", "Test pairs per bucket (probe_test, shared by all seeds):", "",
              "| model | " + " | ".join(f"n({b})" for b in BUCKET_NAMES) + " |", "|---|" + "---|" * len(BUCKET_NAMES)]
    for stem, r in agg["h2"].items():
        lines.append(f"| {stem} | " + " | ".join(_val((r.get("bucket_n") or {}).get(b)) for b in BUCKET_NAMES) + " |")
    lines += ["", "## H3 — belief revision latency", "",
              "| model | mean frac(latency ≤ 5) | supported | frac(≤5), not-visible steps only (exploratory) |",
              "|---|---|---|---|"]
    for stem, r in agg["h3"].items():
        lines.append(f"| {stem} | {r['mean_frac_le5']:.3f} | {r['supported']} | {_fmt(r.get('mean_frac_le5_not_visible'))} |")
    lines += ["", "## H4 — sparsity (k90 = min top-k features reaching 90% of full accuracy)", "",
              "| model | median k90 | #features | strong (≤256) | weak (≤1%) |", "|---|---|---|---|---|"]
    for stem, r in agg["h4"].items():
        lines.append(f"| {stem} | {_fmt(r['median_k90'], 0)} | {r['n_features']} | {r['strong']} | {r['weak']} |")
    lines += ["", "## Prediction quality", "", "| model | params | lr | val CE | test CE | test CE (window) |", "|---|---|---|---|---|---|"]
    for stem, r in agg["perplexity"].items():
        lines.append(f"| {stem} | {_val(r['n_params'])} | {_val(r['lr'])} | {r['val_ce_mean']:.4f} | "
                     f"{_fmt(r.get('test_ce_mean'), 4)} | {_fmt(r.get('test_ce_window_mean'), 4)} |")
    (out / "results.md").write_text("\n".join(lines) + "\n")

    fig, ax = plt.subplots(figsize=(7, 4))
    for stem, r in agg["h2"].items():
        ys = [r["values"].get(b) for b in BUCKET_NAMES]
        ax.plot(range(len(BUCKET_NAMES)), [np.nan if y is None else y for y in ys], marker="o", label=stem)
    ax.set_xticks(range(len(BUCKET_NAMES)), BUCKET_NAMES)
    ax.set_xlabel("steps since last seen")
    ax.set_ylabel("probe accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "h2_curves.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for stem, r in agg["h4"].items():
        for seed, d in r["per_seed"].items():
            ks = sorted(int(k) for k in d["acc_by_k"])
            ax.plot(ks, [d["acc_by_k"][str(k)] for k in ks], marker=".", alpha=0.7,
                    label=f"{stem} (median k90={_fmt(r['median_k90'], 0)})" if seed == min(r["per_seed"]) else None)
    ax.set_xscale("log")
    ax.set_xlabel("k (top features)")
    ax.set_ylabel("accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "h4_curves.png", dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs")
    ap.add_argument("--exp", default="study1")
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    agg = aggregate(args.root, args.exp, args.data)
    write_outputs(agg, args.out or Path(args.root) / args.exp / "results")
    print(json.dumps({"h1": agg["h1"]["supported"], "h2": {k: v["graceful"] for k, v in agg["h2"].items()},
                      "h3": {k: v["supported"] for k, v in agg["h3"].items()},
                      "h4": {k: (v["strong"], v["weak"]) for k, v in agg["h4"].items()}}, indent=2))


if __name__ == "__main__":
    main()
