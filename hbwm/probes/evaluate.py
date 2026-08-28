"""Cross-seed aggregation of probe results; applies the preregistered rules; writes results/."""

import argparse
import json
from pathlib import Path

import numpy as np

from hbwm.matrix import GAMMA_ARMS, best_lr, run_path
from hbwm.probes.decisions import (
    h1_decision,
    h2_curve,
    h3_latency,
    h4_k90,
    h5_decision,
    h6_decision,
    h7_attribution,
    h8_latency,
    is_degenerate,
)
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
    lines += ["", f"## H1: supported = **{h1['supported']}** (margin {h1['margin']:.2f})", "",
              "| comparator | mean diff | paired diffs | passes |", "|---|---|---|---|"]
    for n, c in h1["comparators"].items():
        lines.append(f"| {n} | {c['mean_diff']:+.3f} | {[round(d, 3) for d in c['paired_diffs']]} | {c['passes']} |")
    lines += ["", "## H2: decay curves (accuracy by steps-since-seen bucket)", "",
              "| model | " + " | ".join(BUCKET_NAMES) + " | graceful |", "|---|" + "---|" * (len(BUCKET_NAMES) + 1)]
    for stem, r in agg["h2"].items():
        lines.append(f"| {stem} | " + " | ".join(_fmt(r["values"].get(b)) for b in BUCKET_NAMES) + f" | {r['graceful']} |")
    lines += ["", "Test pairs per bucket (probe_test, shared by all seeds):", "",
              "| model | " + " | ".join(f"n({b})" for b in BUCKET_NAMES) + " |", "|---|" + "---|" * len(BUCKET_NAMES)]
    for stem, r in agg["h2"].items():
        lines.append(f"| {stem} | " + " | ".join(_val((r.get("bucket_n") or {}).get(b)) for b in BUCKET_NAMES) + " |")
    lines += ["", "## H3: belief revision latency", "",
              "| model | mean frac(latency ≤ 5) | supported | frac(≤5), not-visible steps only (exploratory) |",
              "|---|---|---|---|"]
    for stem, r in agg["h3"].items():
        lines.append(f"| {stem} | {r['mean_frac_le5']:.3f} | {r['supported']} | {_fmt(r.get('mean_frac_le5_not_visible'))} |")
    lines += ["", "## H4: sparsity (k90 = min top-k features reaching 90% of full accuracy)", "",
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


STUDY2_MODELS = ["bdh_g100", "lstm", "rwkv"]
STUDY2_LR = 0.003  # RESULTS.md: best_lr.json selected 3e-3 for all three families
STRUCTURED_FAMILIES = ("query_rank_r", "shared_query_rank_r", "derot_query_rank_r",
                       "derot_flat_linear")
MATCHED_FAMILIES = ("flat_linear", "query_rank_r", "shared_query_rank_r", "derot_flat_linear",
                    "derot_query_rank_r", "mlp_rownorm", "mlp_randproj", "mlp_state")


def _study2_val(r):
    return max(r["val_acc"].values())


def _load_study2(root, exp, stem, seed):
    d = Path(root) / exp / f"{stem}_lr{STUDY2_LR:g}" / f"seed{seed}" / "probes2"
    return {p.stem: json.loads(p.read_text()) for p in d.glob("*.json") if p.name != "done.json"}


def _family_key(label, r):
    """Collapse a per-level, per-rank label to its family key: 'query_rank_4' from 'query_rank_4_L3'."""
    return label.split("_L")[0]


def _best_per_family(probes):
    """For each family key, the entry with the highest probe_val accuracy (best level and best l2)."""
    best = {}
    for label, r in probes.items():
        if label.endswith("_bridge"):
            continue
        k = _family_key(label, r)
        if k not in best or _study2_val(r) > _study2_val(best[k][1]):
            best[k] = (label, r)
    return best


def aggregate_study2(root, exp="study2", src_exp="study1", data_dir=None) -> dict:
    per_model = {stem: {seed: _best_per_family(_load_study2(root, exp, stem, seed))
                        for seed in (0, 1, 2)}
                 for stem in STUDY2_MODELS}
    table = {}
    for stem, by_seed in per_model.items():
        keys = set.intersection(*(set(by_seed[s]) for s in by_seed))
        table[stem] = {}
        for k in sorted(keys):
            rows = [by_seed[s][k][1] for s in (0, 1, 2)]
            # Spec 7 degeneracy criterion, applied to the seed-mean accuracies so each arm gets one
            # deterministic verdict. Every arm of every family and model goes through this.
            acc_keys = set(rows[0]["train_acc"])
            tr_mean = {kk: float(np.mean([r["train_acc"][kk] for r in rows])) for kk in acc_keys}
            va_mean = {kk: float(np.mean([r["val_acc"][kk] for r in rows])) for kk in acc_keys}
            deg = is_degenerate(tr_mean, va_mean, rows[0]["chance"])
            best_key = max(va_mean, key=va_mean.get)
            table[stem][k] = {
                "train_acc_mean": tr_mean, "val_acc_mean": va_mean,
                "train_acc_at_best": tr_mean[best_key], "degeneracy": deg,
                "degenerate": deg["degenerate"], "n_input": rows[0].get("n_input"),
                "per_seed": [r["test_acc"] for r in rows],
                "mean": float(np.mean([r["test_acc"] for r in rows])),
                "std": float(np.std([r["test_acc"] for r in rows])),
                "labels": [by_seed[s][k][0] for s in (0, 1, 2)],
                "rank": rows[0]["rank"], "rank_fraction": rows[0]["rank_fraction"],
                "saturated": bool(rows[0]["saturated"]), "n_params": rows[0]["n_params"],
                "n_train": rows[0]["n_train"], "n_features": rows[0]["n_features"],
                "chance": rows[0]["chance"], "ceiling": rows[0]["ceiling"],
                "ci95": [r["ci95"] for r in rows],
                "val_mean": float(np.mean([_study2_val(r) for r in rows])),
            }
    bdh = table["bdh_g100"]
    structured = {k: v for k, v in bdh.items()
                  if any(k.startswith(f.replace("_rank_r", "_rank_")) or k == f
                         for f in STRUCTURED_FAMILIES)}
    best_structured = max(structured, key=lambda k: structured[k]["val_mean"])
    h5 = h5_decision(structured[best_structured]["per_seed"], bdh["flat_linear"]["per_seed"])
    h5["family"] = best_structured
    # Spec 7: a family is eligible for H6 only if it is matched on all three states AND no arm of it,
    # on any state, is degenerate. Degenerate arms stay in `table` and in the results tables.

    def _eligible(k):
        if not all(_alias_key(k, table[m]) for m in ("lstm", "rwkv")):
            return False
        return not (bdh[k]["degenerate"]
                    or any(_arm(table[m], k)["degenerate"] for m in ("lstm", "rwkv")))

    excluded = {k: {"bdh": bdh[k]["degenerate"],
                    **{m: _arm(table[m], k)["degenerate"] for m in ("lstm", "rwkv")}}
                for k in bdh if all(_alias_key(k, table[m]) for m in ("lstm", "rwkv"))
                and not _eligible(k)}
    matched = [k for k in bdh if _eligible(k)]
    if not matched:
        raise RuntimeError(f"no eligible matched family for H6; degeneracy exclusions: {excluded}")
    best_matched = max(matched, key=lambda k: bdh[k]["val_mean"])
    arms = {m: _arm(table[m], best_matched) for m in ("lstm", "rwkv")}
    h6 = h6_decision(bdh[best_matched]["per_seed"], {m: a["per_seed"] for m, a in arms.items()},
                     family=best_matched, saturated={m: a["saturated"] for m, a in arms.items()})
    h6["eligible_families"] = sorted(matched)
    h6["degeneracy_exclusions"] = excluded
    # Spec 4.5 and 7: H7 is the reductions, not the matched `mlp_state` arm.
    mlp_keys = [k for k in ("mlp_rownorm", "mlp_randproj") if k in bdh]
    best_mlp = max(mlp_keys, key=lambda k: bdh[k]["val_mean"]) if mlp_keys else None
    h7 = (h7_attribution(bdh[best_mlp]["per_seed"], structured[best_structured]["per_seed"])
          if best_mlp else {})
    if h7:
        h7["mlp_family"], h7["structured_family"] = best_mlp, best_structured
    bridge = {m: _bridge_rows(root, exp, m) for m in ("lstm", "rwkv")}
    return {"table": table, "h5": h5, "h6": h6, "h7": h7, "h8": _h8_all(root, exp),
            "bridge": bridge, "structure": _structure_all(root, exp)}


def _alias_key(key, other):
    """The baseline arm matched to a BDH family key (spec 5.1), or None if there is none.

    One alias exists: a derot family on a baseline IS its undecorated counterpart (nothing to undo).
    `mlp_state` matches directly because it is defined on every state. `mlp_rownorm` and
    `mlp_randproj` deliberately have NO baseline counterpart (spec 4.5), so they are never matched and
    can never become an H6 arm; they are BDH-only context and feed H7.
    """
    if key in other:
        return key
    if key.startswith("derot_") and key[len("derot_"):] in other:
        return key[len("derot_"):]
    return None


def _arm(other, key):
    return other[_alias_key(key, other)]


def _bridge_rows(root, exp, stem):
    out = {}
    for seed in (0, 1, 2):
        d = (Path(root) / exp / f"{stem}_lr{STUDY2_LR:g}" / f"seed{seed}" / "probes2"
             / "flat_linear_bridge.json")
        if d.exists():
            out[seed] = json.loads(d.read_text())
    return {"per_seed": [out[s]["test_acc"] for s in sorted(out)] if out else [],
            "n_train": (out[min(out)]["n_train"] if out else None), "decides_nothing": True}


def _h8_all(root, exp):
    """The H8 readout of the checkpoint's best spec; `done.json["h8_file"]` names the file."""
    out = {}
    for stem in STUDY2_MODELS:
        rows = []
        for seed in (0, 1, 2):
            d = Path(root) / exp / f"{stem}_lr{STUDY2_LR:g}" / f"seed{seed}" / "probes2"
            done = d / "done.json"
            if not done.exists():
                continue
            name = json.loads(done.read_text()).get("h8_file")
            f = d / name if name else None
            if f is None or not f.exists():
                continue
            a = dict(np.load(f))
            rows.append(h8_latency(a["p_old"], a["p_new"], a["steps_since_reobs"], a["ep"],
                                   a["visible_now"]))
        if rows:
            out[stem] = {"per_seed": rows,
                         "mean_frac_le5": float(np.mean([r["frac_le5"] for r in rows])),
                         "mean_excluded_frac": float(np.mean([r["excluded_frac"] for r in rows])),
                         "low_coverage": any(r["low_coverage"] for r in rows),
                         "supported": bool(np.mean([r["frac_le5"] for r in rows]) >= 0.7)}
    return out


def _structure_all(root, exp):
    out = {}
    for stem in STUDY2_MODELS:
        for seed in (0, 1, 2):
            d = Path(root) / exp / f"{stem}_lr{STUDY2_LR:g}" / f"seed{seed}" / "probes2"
            for f in (sorted(d.glob("sigma_structure_L*.json")) if d.exists() else []):
                out[f"{stem}/seed{seed}/{f.stem}"] = json.loads(f.read_text())
    return out


def write_outputs_study2(agg: dict, out_dir) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for k in ("table", "h5", "h6", "h7", "h8", "bridge", "structure"):
        (out / f"{k}.json").write_text(json.dumps(agg[k], indent=2, default=float) + "\n")
    lines = ["## Study 2 probe accuracy (test, best level and hyperparameter per seed; mean over 3 "
             "seeds)",
             "", "| model | family | rank | eff. rank frac | saturated | degenerate | train acc "
             "| acc | #params | n_train |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for stem, fams in agg["table"].items():
        for k, r in fams.items():
            frac = "-" if r["rank_fraction"] is None else f"{r['rank_fraction']:.2f}"
            lines.append(f"| {stem} | {k} | {_val(r['rank'])} | {frac} | {r['saturated']} "
                         f"| {r['degenerate']} | {r['train_acc_at_best']:.3f} "
                         f"| {r['mean']:.3f} ± {r['std']:.3f} | {_val(r['n_params'])} "
                         f"| {r['n_train']} |")
    d0 = next(iter(next(iter(agg["table"].values())).values()))["degeneracy"]
    lines += ["", f"Degeneracy criterion (preregistered, spec 7): an arm is degenerate, and excluded "
              f"from H6's best-matched-family selection, if its training accuracy exceeds "
              f"{d0['train_bar']:.2f} while its validation accuracy stays below {d0['val_bar']:.3f} "
              f"(twice the majority-class chance rate {d0['chance']:.3f}) at every L2 value. "
              f"Degenerate arms are still fitted and still reported above with their parameter count "
              f"and training accuracy, so the call can be audited. `mlp_state` on BDH has "
              f"268,476,928 parameters against 24,000 training examples, which is the number to read "
              f"that row by."]
    lines += ["", "Family 5 membership: BDH carries `mlp_state` plus the two BDH-only reductions "
              "`mlp_rownorm` and `mlp_randproj`; each baseline carries `mlp_state` alone, because row "
              "norms of a 4 x 350 or 20 x 176 reshape are not a control and projecting 1,400 up to "
              "4,096 is an expansion. H6's family 5 comparison is therefore `mlp_state` against "
              "`mlp_state` against `mlp_state`; the reductions are context and feed H7."]
    h5, h6, h7 = agg["h5"], agg["h6"], agg["h7"]
    lines += ["", f"## H5 (format and estimation) supported: **{h5['supported']}**", "",
              f"Best structured family `{h5['family']}` at {h5['structured_mean']:.3f} against "
              f"`flat_linear` at {h5['flat_mean']:.3f}; mean diff {h5['mean_diff']:+.3f}, "
              f"paired diffs {[round(d, 3) for d in h5['paired_diffs']]}.", "",
              f"## H6 (headline) supported: **{h6['supported']}**, carried by `{h6['family']}`", "",
              "| comparator | mean | mean diff | paired diffs | passes | saturated arm |",
              "|---|---|---|---|---|---|"]
    for name, c in h6["comparators"].items():
        lines.append(f"| {name} | {c['mean']:.3f} | {c['mean_diff']:+.3f} | "
                     f"{[round(d, 3) for d in c['paired_diffs']]} | {c['passes']} "
                     f"| {name in h6['saturated_baselines']} |")
    lines += ["", f"Kill criterion fired: **{h6['kill_criterion_fired']}**. "
              f"Rank-constraint artifact warning: **{h6['artifact_warning']}**.", "",
              f"Families eligible for H6 after the degeneracy criterion: "
              f"{', '.join('`' + f + '`' for f in h6['eligible_families'])}. "
              f"Excluded as degenerate: {h6['degeneracy_exclusions'] or 'none'}.", ""]
    if h7:
        lines += ["## H7 (attribution, gates nothing)", "",
                  f"`{h7['mlp_family']}` at {h7['mlp_mean']:.3f} against `{h7['structured_family']}` "
                  f"at {h7['structured_mean']:.3f}; attribute to capacity: "
                  f"**{h7['attribute_to_capacity']}**.", ""]
    if agg["h8"]:
        lines += ["## H8 (belief revision, clock rebaselined to the first not-visible step)", "",
                  "| model | mean frac(latency <= 5) | mean excluded frac | low coverage | supported |",
                  "|---|---|---|---|---|"]
        for stem, r in agg["h8"].items():
            lines.append(f"| {stem} | {r['mean_frac_le5']:.3f} | {r['mean_excluded_frac']:.3f} "
                         f"| {r['low_coverage']} | {r['supported']} |")
    lines += ["", "## Cross-study bridge rows (continuity only, decide nothing)", "",
              "| model | flat_linear acc at 61,400 pairs |", "|---|---|"]
    for stem, r in agg["bridge"].items():
        lines.append(f"| {stem} | {_fmt(np.mean(r['per_seed']) if r['per_seed'] else None)} |")
    (out / "results.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs")
    ap.add_argument("--exp", default="study1")
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--study", default="1", choices=["1", "2"])
    args = ap.parse_args()
    if args.study == "2":
        agg = aggregate_study2(args.root, args.exp, data_dir=args.data)
        write_outputs_study2(agg, Path(args.root) / args.exp / "results2")
        print(json.dumps({k: agg[k] for k in ("h5", "h6", "h7")}, indent=2, default=float))
        return
    agg = aggregate(args.root, args.exp, args.data)
    write_outputs(agg, args.out or Path(args.root) / args.exp / "results")
    print(json.dumps({"h1": agg["h1"]["supported"], "h2": {k: v["graceful"] for k, v in agg["h2"].items()},
                      "h3": {k: v["supported"] for k, v in agg["h3"].items()},
                      "h4": {k: (v["strong"], v["weak"]) for k, v in agg["h4"].items()}}, indent=2))


if __name__ == "__main__":
    main()
