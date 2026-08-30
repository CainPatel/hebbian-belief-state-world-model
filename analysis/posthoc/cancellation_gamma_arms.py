"""Study 2 post-hoc exploratory analysis: does a LOWER decay gamma actually reduce interference?

EXPLORATORY, POST-HOC, NOT PREREGISTERED. Nothing in this script decides anything. No number here
is a criterion, and none of it can support, refute or revise H1 to H8: those decisions stand
exactly as RESULTS.md records them. This only extends a descriptive quantity that spec section 4.8
measurement 4 defines, from the gamma = 1.0 arm to the two gamma ablation arms Study 1 already
trained.

The question. `analysis/posthoc/cancellation_index.py` found that at gamma = 1.0 BDH's state sigma
destroys roughly 95% of the write mass routed into it: 142 to 214 writes land in each neuron row
per episode and the cancellation index sum(a)/sum(w) sits at 4.9 to 10.7 against a routed mass of
100. That invites an obvious follow-up: would retraining BDH with a better-parameterized decay be
worth the compute? Study 1 already trained `bdh_g099` and `bdh_g097` beside `bdh_g100`, and they
probed WORSE, not better (sigma_full 0.099 and 0.065 against 0.101). But RESULTS.md flags those
arms as CONFOUNDED, because gamma decays PER TOKEN and there are 12 tokens per environment step,
so gamma = 0.99 is about 0.89 per environment step and gamma = 0.97 is about 0.69: the arms may
have been forgetting far too aggressively to say anything about interference.

The cheap decisive test is to measure the cancellation index on the gamma arms themselves.

  Outcome A. Lower gamma RAISES the index (less interference) while probe accuracy still falls.
  Reducing interference then does not buy decodability, and a retraining study with a
  per-environment-step decay is probably not worth the compute.

  Outcome B. Lower gamma does NOT raise the index. The gamma arms then never tested the mechanism
  at all, and a properly parameterized retrain remains open.

Method. Every measurement here reuses `analysis/posthoc/cancellation_index.py` verbatim, by
importing its functions rather than restating them: the same `study2_pairs` draw order, the same
`structure_subsample`, the same `collect` recorder pass with `batch_eps = 32` episode batching
(NOT optional: the un-batched form holds about 12.6 GB of state and this project has already lost
a checkpoint to an OOM kill), `positions=None` so the write accumulator advances at every step, and
the same accumulator ordering its docstring explains. The one thing that changes is the model:
`gamma2 = model.hcfg.decay_gamma ** 2` is no longer 1.0, which that code already handles, so `w` is
now a genuinely decayed sum of routed mass and the ratio stays exponent-matched against sigma.

Levels. Level 3 for both gamma arms, since RESULTS.md reports sigma_full best levels [3, 3, 3] for
each of them. The gamma = 1.0 comparison row is READ from the existing
`runs/study1/results2/posthoc_cancellation.json` rather than recomputed, which is why only its
seed 0 and seed 1 entries appear below: seed 2 of `bdh_g100` was measured at level 4, not level 3.

Read-only on `runs/` and `data/` apart from the one JSON it writes to `runs/study1/results2/`.
"""

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from hbwm.device import release_memory, select_device
from hbwm.envs.tokenizer import STEP_LEN
from hbwm.probes.run import Study2Config
from hbwm.train import load_checkpoint

# The measurement machinery is LOADED from the sibling script, not reimplemented, so that every
# number below comes out of exactly the code that produced posthoc_cancellation.json for gamma = 1.0
# and the two are directly comparable. `analysis/posthoc/` is not a package, hence the file load.
_SIB = Path(__file__).resolve().parent / "cancellation_index.py"
_SPEC = importlib.util.spec_from_file_location("cancellation_index", _SIB)
ci = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ci)

ROOT = ci.ROOT
OUT_DIR = Path(os.environ.get("HBWM_POSTHOC_OUT", ROOT / "runs/study1/results2"))
OUT_NAME = "posthoc_cancellation_gamma.json"
BASELINE_JSON = ROOT / "runs/study1/results2/posthoc_cancellation.json"
TABLE_JSON = ROOT / "runs/study1/results/table.json"

LR_SUFFIX = "_lr0.003"
# RESULTS.md sigma_full best levels are [3, 3, 3] for both gamma arms, so one level covers all seeds.
ARM_LEVELS = {"bdh_g099": 3, "bdh_g097": 3}
BASELINE_STEM = "bdh_g100"
BASELINE_LEVEL = 3  # only seeds 0 and 1 of bdh_g100 were measured at L3; seed 2 was measured at L4


def env_step_gamma(gamma: float) -> float:
    """gamma decays per TOKEN; there are STEP_LEN = 12 tokens per environment step (spec 4.2)."""
    return float(gamma) ** STEP_LEN


def baseline_rows(level: int) -> dict:
    """The already-paid-for gamma = 1.0 rows, read back rather than recomputed."""
    if not BASELINE_JSON.exists():
        return {}
    d = json.loads(BASELINE_JSON.read_text())
    out = {}
    for k, v in d.items():
        if v.get("run", "").startswith(BASELINE_STEM) and v.get("level") == level:
            out[int(v["seed"])] = v
    return out


def probe_acc(stem: str) -> dict:
    """Study 1's sigma_full probe accuracy for one stem, from runs/study1/results/table.json."""
    if not TABLE_JSON.exists():
        return {}
    t = json.loads(TABLE_JSON.read_text()).get(stem, {}).get("sigma_full", {})
    return {"per_seed": t.get("per_seed"), "mean": t.get("mean"), "std": t.get("std"),
            "levels": t.get("levels"), "chance": t.get("chance")}


def row_summary(r: dict) -> dict:
    """The reported fields, pulled out of a full `cancellation_stats` payload."""
    return {
        "index_median": r["index_pooled"]["median"],
        "index_p10": r["index_pooled"]["p10"],
        "index_p90": r["index_pooled"]["p90"],
        "index_top_1pct": r["index_top_1pct"]["median"],
        "index_top_10pct": r["index_top_10pct"]["median"],
        "writes_per_row_median": r["writes_per_row_median"]["median"],
        "writes_per_row_top_1pct": r["writes_per_row_top_1pct"]["median"],
        "writes_per_row_top_10pct": r["writes_per_row_top_10pct"]["median"],
        "writes_per_row_max": r["writes_per_row_max"]["median"],
        "corr_w_a_pearson": r["corr_w_a_pearson"]["median"],
        "corr_w_a_spearman": r["corr_w_a_spearman"]["median"],
        "frac_rows_never_written": r["frac_rows_never_written"]["median"],
        "decay_gamma": r.get("decay_gamma"),
        "level": r.get("level"),
        "seed": r.get("seed"),
        "n_examples": r.get("n_examples"),
    }


def reading(by_stem: dict, base: dict, accs: dict, matched_seeds: list) -> str:
    """Outcome A or outcome B, decided from the numbers rather than asserted."""
    if not base or not matched_seeds:
        return ("No gamma = 1.0 comparison row at this level, so neither outcome can be read off. "
                "Run `cancellation_index.py` for the matching seeds and level first.")
    b = float(np.mean([base[s]["index_pooled"]["median"] for s in matched_seeds]))
    lines = []
    verdicts = []
    for stem, rows in by_stem.items():
        have = [s for s in matched_seeds if s in rows]
        if not have:
            continue
        g = float(np.mean([rows[s]["index_median"] for s in have]))
        wr_b = float(np.mean([base[s]["writes_per_row_median"]["median"] for s in matched_seeds]))
        wr_g = float(np.mean([rows[s]["writes_per_row_median"] for s in have]))
        acc = accs.get(stem, {}).get("mean")
        acc0 = accs.get(BASELINE_STEM, {}).get("mean")
        raised = g > b
        verdicts.append(raised)
        lines.append(
            f"{stem} (gamma = {rows[have[0]]['decay_gamma']:.2f} per token, "
            f"{env_step_gamma(rows[have[0]]['decay_gamma']):.3f} per environment step): "
            f"cancellation index {g:.2f} against {b:.2f} at gamma = 1.0 on the same seeds "
            f"{have}, a {'RISE' if raised else 'FALL'} of {abs(g - b):.2f}; writes per row "
            f"{wr_g:.0f} against {wr_b:.0f}; Study 1 sigma_full accuracy "
            f"{acc:.3f} against {acc0:.3f}." if acc is not None and acc0 is not None else
            f"{stem}: cancellation index {g:.2f} against {b:.2f}.")
    if not verdicts:
        return "No gamma arm rows computed, so neither outcome can be read off."
    if all(verdicts):
        head = ("OUTCOME A obtained. Lower gamma DOES raise the cancellation index, so the gamma "
                "arms really did reduce interference, and probe accuracy still did not improve "
                "(it fell). Reducing interference therefore does not buy decodability on its own, "
                "and a retraining study with a per-environment-step decay looks like poor value "
                "for the compute: the mechanism it would target has now been varied without the "
                "readout following.")
    elif not any(verdicts):
        head = ("OUTCOME B obtained. Lower gamma does NOT raise the cancellation index, so the "
                "gamma arms never tested the interference mechanism at all and their lower probe "
                "accuracy says nothing about it. The confound RESULTS.md flags is real and "
                "unresolved, and a retrain with a decay parameterized per environment step rather "
                "than per token remains an open and still-motivated question.")
    else:
        head = ("MIXED. The two gamma arms disagree on whether the index rises, so neither outcome "
                "obtains cleanly and the per-arm rows below have to be read individually.")
    tail = ("None of this revises H1 to H8, and none of it is a preregistered comparison: the "
            "gamma arms were trained under a per-token decay, so any reading here describes those "
            "checkpoints as trained, not a controlled manipulation of interference.")
    return head + " " + " ".join(lines) + " " + tail


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stems", nargs="*", default=list(ARM_LEVELS))
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--levels", type=int, nargs="*", default=None,
                    help="override the per-arm level (default 3, RESULTS.md's sigma_full best)")
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
    # The SAME pair draw as gamma = 1.0: study2_pairs reproduces all four generator calls in order,
    # then structure_subsample takes measure_sigma_structure's own seeded 1,024-pair subsample.
    data, p_tr = ci.study2_pairs(cfg)
    p = ci.structure_subsample(p_tr, args.n_sample, cfg.seed)
    print(f"device={device} probe_train pairs={len(p_tr)} sampled={len(p)} "
          f"episodes={len(np.unique(p.ep))} batch_eps={args.batch_eps} T={data.T} "
          f"({time.time() - t0:.1f}s to build pairs)")

    refs_path = ROOT / "runs/study1/results2/structure.json"
    refs = json.loads(refs_path.read_text()) if refs_path.exists() else {}
    out, rows, by_stem = {}, [], {}
    for stem in args.stems:
        by_stem[stem] = {}
        levels = args.levels if args.levels is not None else [ARM_LEVELS[stem]]
        run = f"{stem}{LR_SUFFIX}"
        for seed in args.seeds:
            ckpt = ROOT / "runs/study1" / run / f"seed{seed}/ckpt.pt"
            if not ckpt.exists():
                print(f"  skip {stem}/seed{seed}: no checkpoint at {ckpt}")
                continue
            model, _, meta = load_checkpoint(ckpt, device)
            t1 = time.time()
            got = ci.collect(model, data, p, levels, device, args.batch_eps)
            secs = time.time() - t1
            for lvl in levels:
                norms, wmass, nz, pos = got[lvl]
                r = ci.cancellation_stats(norms, wmass, nz, pos, data.T)
                r["crosscheck"] = ci.crosscheck(
                    norms, wmass,
                    refs.get(f"{stem}/seed{seed}/sigma_structure_L{lvl}")
                    if args.n_sample == Study2Config.structure_n_sample else None)
                r.update({"level": lvl, "seed": seed, "run": run, "ckpt": str(ckpt),
                          "ckpt_step": int(meta["step"]),
                          "decay_gamma": float(model.hcfg.decay_gamma),
                          "decay_gamma_per_env_step": env_step_gamma(model.hcfg.decay_gamma),
                          "tokens_per_env_step": int(STEP_LEN),
                          "n_sample": args.n_sample, "sample_seed": cfg.seed,
                          "batch_eps": args.batch_eps, "elapsed_s": round(secs, 1),
                          "exploratory": True, "preregistered": False})
                key = f"{stem}/seed{seed}/cancellation_L{lvl}"
                out[key] = r
                rows.append((key, r))
                if lvl == BASELINE_LEVEL:
                    by_stem[stem][seed] = row_summary(r)
            del got, model
            release_memory(device)
            print(f"  {stem}/seed{seed} levels={levels}: {secs:.1f}s")

    base = baseline_rows(BASELINE_LEVEL)
    accs = {s: probe_acc(s) for s in [BASELINE_STEM, *args.stems]}
    matched = sorted(set(base) & {s for v in by_stem.values() for s in v})

    hdr = (f"{'arm/seed-level':30s} {'gamma':>6s} {'g^12':>6s} {'index (med [p10,p90])':>26s} "
           f"{'top1%':>7s} {'top10%':>7s} {'r(w,a)':>7s} {'rho':>6s} {'wr/row':>7s} {'never':>6s}")
    print("\n=== cancellation index  sum(a)/sum(w)  (<1 writes cancel, >1 they reinforce) ===")
    print(hdr)
    print("-" * len(hdr))
    for seed in sorted(base):
        r = base[seed]
        i = r["index_pooled"]
        print(f"{'bdh_g100/seed' + str(seed) + '-L' + str(r['level']) + ' (read)':30s} "
              f"{1.0:6.2f} {1.0:6.3f} "
              + f"{i['median']:.4f} [{i['p10']:.4f},{i['p90']:.4f}]".rjust(26)
              + f" {r['index_top_1pct']['median']:7.4f} {r['index_top_10pct']['median']:7.4f} "
                f"{r['corr_w_a_pearson']['median']:7.3f} "
                f"{r['corr_w_a_spearman']['median']:6.3f} "
                f"{r['writes_per_row_median']['median']:7.1f} "
                f"{r['frac_rows_never_written']['median']:6.3f}")
    for key, r in rows:
        i = r["index_pooled"]
        print(f"{key.replace('/cancellation_L', '-L'):30s} {r['decay_gamma']:6.2f} "
              f"{r['decay_gamma_per_env_step']:6.3f} "
              + f"{i['median']:.4f} [{i['p10']:.4f},{i['p90']:.4f}]".rjust(26)
              + f" {r['index_top_1pct']['median']:7.4f} {r['index_top_10pct']['median']:7.4f} "
                f"{r['corr_w_a_pearson']['median']:7.3f} "
                f"{r['corr_w_a_spearman']['median']:6.3f} "
                f"{r['writes_per_row_median']['median']:7.1f} "
                f"{r['frac_rows_never_written']['median']:6.3f}")
    for key, r in rows:
        c = r["crosscheck"]
        print(f"{key}: sum_a med={r['sum_a']['median']:.4g} sum_w med={r['sum_w']['median']:.4g} "
              f"finite={r['all_finite']} positive={r['all_positive']} "
              f"reproduces structure.json={c['reproduced_structure_json']}")

    print("\n=== across gamma, with Study 1's sigma_full probe accuracy beside it "
          f"(level {BASELINE_LEVEL}, seeds {matched or 'n/a'}) ===")
    sh = (f"{'arm':10s} {'gamma/tok':>9s} {'gamma/step':>10s} {'index':>8s} {'wr/row':>7s} "
          f"{'sigma_full acc':>16s} {'levels':>10s}")
    print(sh)
    print("-" * len(sh))

    def _acc_cell(stem):
        a = accs.get(stem, {})
        return (f"{a['mean']:.3f} +- {a['std']:.3f}" if a.get("mean") is not None else "n/a")

    if base:
        bi = float(np.mean([base[s]["index_pooled"]["median"] for s in (matched or sorted(base))]))
        bw = float(np.mean([base[s]["writes_per_row_median"]["median"]
                            for s in (matched or sorted(base))]))
        print(f"{BASELINE_STEM:10s} {1.0:9.2f} {1.0:10.3f} {bi:8.3f} {bw:7.1f} "
              f"{_acc_cell(BASELINE_STEM):>16s} "
              f"{str(accs.get(BASELINE_STEM, {}).get('levels')):>10s}")
    for stem, srows in by_stem.items():
        have = [s for s in (matched or sorted(srows)) if s in srows]
        if not have:
            continue
        g = srows[have[0]]["decay_gamma"]
        print(f"{stem:10s} {g:9.2f} {env_step_gamma(g):10.3f} "
              f"{float(np.mean([srows[s]['index_median'] for s in have])):8.3f} "
              f"{float(np.mean([srows[s]['writes_per_row_median'] for s in have])):7.1f} "
              f"{_acc_cell(stem):>16s} {str(accs.get(stem, {}).get('levels')):>10s}")

    text = reading(by_stem, base, accs, matched)
    print("\n=== reading ===")
    print(text)

    out["summary"] = {
        "level": BASELINE_LEVEL,
        "matched_seeds": matched,
        "baseline_stem": BASELINE_STEM,
        "baseline_index_median_by_seed": {str(s): base[s]["index_pooled"]["median"]
                                          for s in sorted(base)},
        "baseline_source": str(BASELINE_JSON),
        "arm_index_median_by_seed": {stem: {str(s): v["index_median"] for s, v in srows.items()}
                                     for stem, srows in by_stem.items()},
        "probe_acc_sigma_full": accs,
        "tokens_per_env_step": int(STEP_LEN),
        "reading": text,
        "exploratory": True,
        "preregistered": False,
    }

    if not args.no_write:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Merge by default: one checkpoint-level is a long recorder pass, so a run covering a
        # subset of seeds must not discard keys an earlier run already paid for.
        merged = {}
        if path.exists() and not args.overwrite:
            merged = json.loads(path.read_text())
        merged.update(out)
        path.write_text(json.dumps(merged, indent=2) + "\n")
        print(f"\nwrote {path} ({len(merged)} keys)")
    print(f"total wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    with torch.no_grad():
        main()
