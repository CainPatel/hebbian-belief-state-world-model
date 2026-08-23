"""Per-checkpoint probe runner (spec section 4.5). Writes <run_dir>/probes/."""

import argparse
import dataclasses
import json
import shutil
import time
from pathlib import Path

import numpy as np

from hbwm.bdh.core import HBWMCore
from hbwm.config import to_dict
from hbwm.device import select_device
from hbwm.envs.dataset import EpisodeData
from hbwm.instrument.atlas import build_atlas, save_atlas
from hbwm.instrument.features import feature_dim, n_levels, neuron_of_feature
from hbwm.probes.eligibility import BUCKET_NAMES, PairSet, h3_pairs, sample_pairs
from hbwm.probes.extract import collect_many, iter_features
from hbwm.probes.probe import (
    accuracy,
    bootstrap_ci,
    majority_chance,
    predict_proba_stream,
    train_probes_multi,
)
from hbwm.train import load_checkpoint


def _l(x):
    return list(x)


@dataclasses.dataclass
class ProbeConfig:
    small_features: list = dataclasses.field(default_factory=lambda: ["sigma_rownorm", "x_sparse", "resid"])
    full_feature: str = "sigma_full"
    full_levels: str = "auto"
    per_obj: int = 8
    l2_grid: list = dataclasses.field(default_factory=lambda: [1e-4, 1e-3, 1e-2, 1e-1])
    epochs: int = 20
    lr: float = 1e-3
    batch: int = 512
    n_train_full: int = 24000
    batch_eps: int = 64
    seed: int = 0
    h3: bool = True
    h4: bool = True
    h4_ks: list = dataclasses.field(default_factory=lambda: [16, 64, 256, 1024, 4096, 16384])
    atlas: bool = True
    atlas_episodes: int = 500
    n_boot: int = 1000


PRESETS = {
    "study1": ProbeConfig(),
    "smoke": ProbeConfig(small_features=["sigma_rownorm"], per_obj=2, l2_grid=[1e-3], epochs=2, n_train_full=50,
                         batch_eps=4, h4_ks=[2, 4], atlas_episodes=4, n_boot=20),
}


def spec_name(spec) -> str:
    f, lvl = spec
    return f if lvl is None else f"{f}_L{lvl}"


def val_best(r: dict) -> float:
    return r["val_acc"][f"{r['best_l2']:g}"]


def stratified_subsample(pairs: PairSet, n_max: int, rng) -> PairSet:
    if len(pairs) <= n_max:
        return pairs
    groups = {b: _l(rng.permutation(np.where(pairs.bucket == b)[0])) for b in np.unique(pairs.bucket)}
    chosen = []
    while len(chosen) < n_max:
        for b in sorted(groups):
            if groups[b] and len(chosen) < n_max:
                chosen.append(groups[b].pop())
    return pairs.subset(np.sort(np.array(chosen)))


def gather_columns(X, cols, chunk=2048) -> np.ndarray:
    out = np.empty((X.shape[0], len(cols)), dtype=np.float32)
    for b0 in range(0, X.shape[0], chunk):
        out[b0 : b0 + chunk] = np.asarray(X[b0 : b0 + chunk][:, cols], dtype=np.float32)
    return out


def fit_and_eval_stage(model, data, pairs, specs, pcfg, n_classes, device, out_dir, cache_dir, memmap_specs,
                       select_h34, h3p, chance, ceiling):
    """data = (d_tr, d_va, d_te); pairs = (p_tr, p_va, p_te). Trains + selects on val + evaluates on test,
    runs H4 top-k retrains and H3 readouts for the specs returned by select_h34(results)."""
    d_tr, d_va, d_te = data
    p_tr, p_va, p_te = pairs
    dims = {s: feature_dim(model, s[0]) for s in specs}
    X = collect_many(iter_features(model, d_tr, p_tr, specs, pcfg.batch_eps, device), len(p_tr), dims,
                     np.float32, cache_dir, memmap_specs)
    probes = {s: train_probes_multi(X[s], p_tr.label, n_classes, pcfg.l2_grid, pcfg.epochs, pcfg.lr,
                                    pcfg.batch, pcfg.seed, device) for s in specs}
    val = predict_proba_stream(probes, iter_features(model, d_va, p_va, specs, pcfg.batch_eps, device),
                               len(p_va), n_classes, device)
    results, best = {}, {}
    for s in specs:
        val_acc = {l2: accuracy(val[s][l2], p_va.label) for l2 in pcfg.l2_grid}
        best[s] = max(val_acc, key=val_acc.get)
        results[s] = {"feature": s[0], "level": s[1], "n_features": dims[s], "n_train": len(p_tr),
                      "n_val": len(p_va), "n_test": len(p_te), "val_acc": {f"{k:g}": v for k, v in val_acc.items()},
                      "best_l2": best[s], "chance": chance, "ceiling": ceiling}
    sel = select_h34(results) if select_h34 else []
    test_probes = {s: {"best": probes[s][best[s]]} for s in specs}
    columns, h4_rank = {}, {}
    if pcfg.h4:
        for s in sel:
            W = probes[s][best[s]].linear.weight.detach().cpu().numpy()
            rank = np.argsort(-np.linalg.norm(W, axis=0))
            ks = [k for k in pcfg.h4_ks if k < dims[s]]
            if not ks:
                continue
            top = rank[: max(ks)]
            Xk = gather_columns(X[s], top)
            h4_rank[s] = (rank, ks)
            columns[s] = {}
            for k in ks:
                pk = train_probes_multi(Xk[:, :k], p_tr.label, n_classes, [best[s]], pcfg.epochs, pcfg.lr,
                                        pcfg.batch, pcfg.seed, device)[best[s]]
                test_probes[s][f"k{k}"] = pk
                columns[s][f"k{k}"] = top[:k]
    test = predict_proba_stream(test_probes, iter_features(model, d_te, p_te, specs, pcfg.batch_eps, device),
                                len(p_te), n_classes, device, columns)
    for s in specs:
        probs = test[s]["best"]
        correct = probs.argmax(1) == p_te.label
        r = results[s]
        r["test_acc"] = float(correct.mean())
        r["ci95"] = list(bootstrap_ci(correct, p_te.ep, pcfg.n_boot))
        r["bucket_acc"] = {BUCKET_NAMES[b]: (float(correct[p_te.bucket == b].mean()) if (p_te.bucket == b).any() else None)
                           for b in range(len(BUCKET_NAMES))}
        r["bucket_n"] = {BUCKET_NAMES[b]: int((p_te.bucket == b).sum()) for b in range(len(BUCKET_NAMES))}
        np.savez(out_dir / f"{spec_name(s)}_test.npz", probs=probs.astype(np.float16), label=p_te.label, ep=p_te.ep,
                 t=p_te.t, obj=p_te.obj, bucket=p_te.bucket, oracle=p_te.oracle)
        if s in h4_rank:
            rank, ks = h4_rank[s]
            is_neuron_feature = isinstance(model, HBWMCore) and s[0] in ("sigma_full", "sigma_rownorm", "x_sparse")
            # "l2" is the full-feature best_l2 that every top-k probe was retrained with; L2 is *not*
            # re-selected per k, so acc_by_k isolates the effect of the feature budget alone.
            r["h4"] = {
                "ks": ks,
                "l2": best[s],
                "acc_by_k": {str(k): accuracy(test[s][f"k{k}"], p_te.label) for k in ks},
                "acc_all": r["test_acc"],
                "neurons_by_k": ({str(k): int(len(np.unique(neuron_of_feature(model.hcfg, s[0], rank[:k])))) for k in ks}
                                 if is_neuron_feature else None),
                "n_neurons_total": (model.hcfg.n_head * model.hcfg.n_neurons if is_neuron_feature else None),
            }
        (out_dir / f"{spec_name(s)}.json").write_text(json.dumps(r, indent=2) + "\n")
    if pcfg.h3 and h3p is not None and len(h3p) and sel:
        h3 = predict_proba_stream({s: {"best": probes[s][best[s]]} for s in sel},
                                  iter_features(model, d_te, h3p, sel, pcfg.batch_eps, device), len(h3p), n_classes, device)
        rows = np.arange(len(h3p))
        for s in sel:
            probs = h3[s]["best"]
            np.savez(out_dir / f"{spec_name(s)}_h3.npz", p_old=probs[rows, h3p.old_cell], p_new=probs[rows, h3p.new_cell],
                     ep=h3p.ep, t=h3p.t, steps_since_reobs=h3p.steps_since_reobs, visible_now=h3p.visible_now)
    return results


def run_probes(run_dir, data_dir, pcfg: ProbeConfig, device=None) -> dict:
    run_dir = Path(run_dir)
    out = run_dir / "probes"
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "cache"
    try:
        device = select_device(device)
        t0 = time.time()
        model, tcfg, meta = load_checkpoint(run_dir / "ckpt.pt", device)
        d_tr, d_va, d_te = (EpisodeData(data_dir, s) for s in ("probe_train", "probe_val", "probe_test"))
        rng = np.random.default_rng(pcfg.seed)
        p_tr, p_va, p_te = (sample_pairs(d, rng, pcfg.per_obj) for d in (d_tr, d_va, d_te))
        for name, p in (("train", p_tr), ("val", p_va), ("test", p_te)):
            p.save(out / f"pairs_{name}.npz")
        n_classes = d_tr.G * d_tr.G
        chance = majority_chance(p_tr.label, p_te.label)
        ceiling = float((p_te.oracle == p_te.label).mean())
        h3p = h3_pairs(d_te)
        summary = {"chance": chance, "ceiling": ceiling, "n_classes": n_classes, "n_h3": int(len(h3p)), "specs": []}
        data, pairs = (d_tr, d_va, d_te), (p_tr, p_va, p_te)
        all_results = {}
        if isinstance(model, HBWMCore):
            levels = list(range(n_levels(model)))
            specs_small = [(f, lvl) for f in pcfg.small_features for lvl in levels]
            res_small = fit_and_eval_stage(model, data, pairs, specs_small, pcfg, n_classes, device, out, cache, (),
                                           None, None, chance, ceiling)
            all_results.update(res_small)
            if pcfg.full_feature:
                if pcfg.full_levels == "all" or (pcfg.full_levels == "auto" and "sigma_rownorm" not in pcfg.small_features):
                    lv = levels
                elif pcfg.full_levels == "auto":
                    rn = {lvl: val_best(res_small[("sigma_rownorm", lvl)]) for lvl in levels}
                    lv = sorted(set(sorted(rn, key=rn.get, reverse=True)[:2]) | {levels[-1]})
                else:  # explicit "3,1": de-duplicate and sort so the lowest-level val tie-break holds
                    lv = sorted({int(x) for x in pcfg.full_levels.split(",")})
                    if not set(lv) <= set(levels):
                        raise ValueError(f"full_levels {pcfg.full_levels!r} outside range({len(levels)})")
                specs_full = [(pcfg.full_feature, lvl) for lvl in lv]
                p_tr_full = stratified_subsample(p_tr, pcfg.n_train_full, rng)
                p_tr_full.save(out / "pairs_train_full.npz")

                def select_best_full(results):
                    return [max(specs_full, key=lambda s: val_best(results[s]))]

                res_full = fit_and_eval_stage(model, data, (p_tr_full, p_va, p_te), specs_full, pcfg, n_classes, device,
                                              out, cache, tuple(specs_full), select_best_full, h3p, chance, ceiling)
                all_results.update(res_full)
                summary["best_full_spec"] = spec_name(select_best_full(res_full)[0])
            if pcfg.atlas:
                save_atlas(build_atlas(model, d_tr, pcfg.atlas_episodes, device=device), out / "atlas.json")
        else:
            specs = [("state_vec", None)]
            all_results.update(fit_and_eval_stage(model, data, pairs, specs, pcfg, n_classes, device, out, cache, (),
                                                  lambda r: specs, h3p, chance, ceiling))
    finally:  # the fp16 sigma_full memmaps are ~25 GB per level in Study 1 and must not survive a failure
        shutil.rmtree(cache, ignore_errors=True)
    summary["specs"] = sorted(spec_name(s) for s in all_results)
    summary["elapsed_s"] = round(time.time() - t0, 1)
    summary["probe_cfg"] = to_dict(pcfg)
    (out / "done.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--data", default="data/grid9")
    ap.add_argument("--preset", default="study1", choices=sorted(PRESETS))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    print(json.dumps(run_probes(args.run_dir, args.data, PRESETS[args.preset], args.device)))


if __name__ == "__main__":
    main()
