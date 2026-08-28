"""Per-checkpoint probe runner (spec section 4.5). Writes <run_dir>/probes/."""

import argparse
import dataclasses
import json
import resource
import shutil
import time
from pathlib import Path

import numpy as np

from hbwm.bdh.core import HBWMCore
from hbwm.config import to_dict
from hbwm.device import release_memory, select_device
from hbwm.envs import tokenizer as tk
from hbwm.envs.dataset import EpisodeData
from hbwm.instrument.atlas import build_atlas, save_atlas
from hbwm.instrument.features import feature_dim, n_levels, neuron_of_feature
from hbwm.probes.eligibility import BUCKET_NAMES, PairSet, h3_pairs, sample_pairs
from hbwm.probes.extract import collect_many, iter_features
from hbwm.probes.probe import (
    accuracy,
    bootstrap_ci,
    feature_stats,
    majority_chance,
    predict_proba_stream,
    train_probes_multi,
)
from hbwm.probes.structured import (
    FamilySpec,
    apply_randproj,
    evaluate_on,
    family_specs,
    param_count,
    sparse_randproj,
    spec_label,
    state_shape,
    train_family_probes,
)
from hbwm.train import load_checkpoint


def _l(x):
    return list(x)


def _log_mem(tag: str) -> None:
    """One `[mem]` line per stage boundary, for the matrix log. On macOS ru_maxrss is bytes."""
    print(f"[mem] {tag} maxrss_gb={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.1f}", flush=True)


def _stage_boundary(tag: str, device=None) -> None:
    """Between recorder passes: return what the finished stage no longer holds, then log the high-water
    mark. Purely hygiene: nothing still referenced is touched, so results are unaffected."""
    release_memory(device)
    _log_mem(tag)


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
    # batch_eps=32 (not the 64 default): smaller recorder passes bound the peak device footprint of a
    # Study 1 checkpoint. Grouping only: every pair is still extracted exactly once, and each row is
    # written back by index, so the features and every downstream number are unchanged.
    "study1": ProbeConfig(batch_eps=32),
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
                       select_h34, h3p, chance, ceiling, stage="stage"):
    """data = (d_tr, d_va, d_te); pairs = (p_tr, p_va, p_te). Trains + selects on val + evaluates on test,
    runs H4 top-k retrains and H3 readouts for the specs returned by select_h34(results). `stage` only
    names this stage in the `[mem]` log lines."""
    d_tr, d_va, d_te = data
    p_tr, p_va, p_te = pairs
    dims = {s: feature_dim(model, s[0]) for s in specs}
    X = collect_many(iter_features(model, d_tr, p_tr, specs, pcfg.batch_eps, device), len(p_tr), dims,
                     np.float32, cache_dir, memmap_specs)
    probes = {s: train_probes_multi(X[s], p_tr.label, n_classes, pcfg.l2_grid, pcfg.epochs, pcfg.lr,
                                    pcfg.batch, pcfg.seed, device) for s in specs}
    _stage_boundary(f"{stage}:train", device)
    val = predict_proba_stream(probes, iter_features(model, d_va, p_va, specs, pcfg.batch_eps, device),
                               len(p_va), n_classes, device)
    results, best = {}, {}
    for s in specs:
        val_acc = {l2: accuracy(val[s][l2], p_va.label) for l2 in pcfg.l2_grid}
        best[s] = max(val_acc, key=val_acc.get)
        results[s] = {"feature": s[0], "level": s[1], "n_features": dims[s], "n_train": len(p_tr),
                      "n_val": len(p_va), "n_test": len(p_te), "val_acc": {f"{k:g}": v for k, v in val_acc.items()},
                      "best_l2": best[s], "chance": chance, "ceiling": ceiling}
    del val  # every val probability the run needs is already summarised in results[s]["val_acc"]
    _stage_boundary(f"{stage}:val", device)
    sel = select_h34(results) if select_h34 else []
    test_probes = {s: {"best": probes[s][best[s]]} for s in specs}
    del probes  # the selected probes are those same objects; the unselected L2s are dead weight
    columns, h4_rank = {}, {}
    if pcfg.h4:
        for s in sel:
            W = test_probes[s]["best"].linear.weight.detach().cpu().numpy()
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
            del Xk  # up to max(h4_ks) fp32 columns of the training set, per selected spec
            release_memory(device)
    # Last use of X is the gather_columns above: drop the ~24 GB fp32 cache and close the fp16 memmaps
    # before the test and H3 passes push more through the recorder.
    del X
    _stage_boundary(f"{stage}:h4", device)
    test = predict_proba_stream(test_probes, iter_features(model, d_te, p_te, specs, pcfg.batch_eps, device),
                                len(p_te), n_classes, device, columns)
    _stage_boundary(f"{stage}:test", device)
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
    del test  # written out above (probs as fp16 npz); the H3 pass builds its own
    if pcfg.h3 and h3p is not None and len(h3p) and sel:
        h3 = predict_proba_stream({s: {"best": test_probes[s]["best"]} for s in sel},
                                  iter_features(model, d_te, h3p, sel, pcfg.batch_eps, device), len(h3p), n_classes, device)
        rows = np.arange(len(h3p))
        for s in sel:
            probs = h3[s]["best"]
            np.savez(out_dir / f"{spec_name(s)}_h3.npz", p_old=probs[rows, h3p.old_cell], p_new=probs[rows, h3p.new_cell],
                     ep=h3p.ep, t=h3p.t, steps_since_reobs=h3p.steps_since_reobs, visible_now=h3p.visible_now)
        del h3
    _stage_boundary(f"{stage}:h3", device)
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
                                           None, None, chance, ceiling, stage="small")
            all_results.update(res_small)
            _stage_boundary("small:done", device)
            if pcfg.full_feature:
                if pcfg.full_levels == "auto" and "sigma_rownorm" not in pcfg.small_features:
                    # silently probing every level instead would blow the sigma_full budget (~25 GB/level)
                    raise ValueError("full_levels='auto' requires sigma_rownorm in small_features; "
                                     "use 'all' or an explicit list")
                if pcfg.full_levels == "all":
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
                                              out, cache, tuple(specs_full), select_best_full, h3p, chance, ceiling,
                                              stage="full")
                all_results.update(res_full)
                summary["best_full_spec"] = spec_name(select_best_full(res_full)[0])
                _stage_boundary("full:done", device)
            if pcfg.atlas:
                try:  # exploratory: an atlas failure must never cost the preregistered probe results
                    save_atlas(build_atlas(model, d_tr, pcfg.atlas_episodes, device=device), out / "atlas.json")
                except Exception as e:
                    summary["atlas_error"] = repr(e)
                _stage_boundary("atlas:done", device)
        else:
            specs = [("state_vec", None)]
            all_results.update(fit_and_eval_stage(model, data, pairs, specs, pcfg, n_classes, device, out, cache, (),
                                                  lambda r: specs, h3p, chance, ceiling, stage="baseline"))
    finally:  # the fp16 sigma_full memmaps are ~25 GB per level in Study 1 and must not survive a failure
        shutil.rmtree(cache, ignore_errors=True)
    summary["specs"] = sorted(spec_name(s) for s in all_results)
    summary["elapsed_s"] = round(time.time() - t0, 1)
    summary["probe_cfg"] = to_dict(pcfg)
    (out / "done.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


@dataclasses.dataclass
class Study2Config:
    """Spec sections 4.7, 5.1, 6. Defaults are the preregistered Study 2 values."""

    families: list = dataclasses.field(default_factory=list)  # empty = the full preregistered set
    levels: list = dataclasses.field(default_factory=list)  # BDH levels; [None] for a baseline
    l2_grid: list = dataclasses.field(default_factory=lambda: [1e-4, 1e-3, 1e-2, 1e-1])
    epochs: int = 20
    lr: float = 1e-3
    batch: int = 512
    n_restarts: int = 3
    n_train: int = 24000
    n_train_bridge: int = 61400
    per_obj: int = 8
    batch_eps: int = 32
    seed: int = 0
    n_boot: int = 1000
    randproj_dim: int = 4096
    randproj_density: int = 64
    bridge: bool = True
    structure: bool = True  # exploratory spec 4.8 block; wired up in Task 12
    structure_n_sample: int = 1024


RANDPROJ_CHUNK = 256  # the [chunk, n_out, density] temporary is 256 * 4096 * 64 floats = 268 MB


def _derived_inputs(X_flat, X_rownorm, kind, proj, flat_stats=None):
    if kind in ("flat", "state"):
        return X_flat
    if kind == "rownorm":
        return X_rownorm
    idx, sign = proj
    # Spec 4.5: the control is a fixed sparse projection of the STANDARDIZED flat sigma. `flat_stats`
    # is always the TRAIN split's (mean, std) -- on val, test and the H8 pass too -- so the probe is
    # fitted on and streamed over one input distribution. Projecting the raw fp16 row instead would
    # make the control a Johnson-Lindenstrauss projection dominated by whichever sigma entries happen
    # to carry the largest raw variance, which is not the preregistered arm.
    if flat_stats is None:
        raise ValueError("randproj needs the train-split flat statistics (spec 4.5)")
    mean, std = flat_stats
    out = np.empty((X_flat.shape[0], idx.shape[0]), dtype=np.float32)
    for b0 in range(0, X_flat.shape[0], RANDPROJ_CHUNK):  # X_flat may be the 25 GB fp16 memmap
        chunk = X_flat[b0 : b0 + RANDPROJ_CHUNK]
        z = (chunk.astype(np.float32) - mean) / std
        out[b0 : b0 + RANDPROJ_CHUNK] = apply_randproj(z, idx, sign)
    return out


def _study2_selection(cfg: Study2Config, model, shape, is_bdh):
    """Validate `--families`/`--levels` and return (specs, levels), matching Study 1's `full_levels`.

    Every check here happens BEFORE the first recorder pass. A typo'd family would otherwise leave
    `specs` empty, write a ~25 GB `sigma_full_L{n}.npy` cache, train nothing and die hours later on
    `max()` of an empty sequence; a negative level would index `payload["sigma"][-1]`, silently
    scoring the LAST level while every artifact is labelled `_L-1`.
    """
    known = {spec_label(s) for s in family_specs(shape)}
    unknown = sorted(set(cfg.families) - known)
    if unknown:
        raise ValueError(f"unknown families {unknown}; this model's families are {sorted(known)}")
    if not is_bdh:  # a baseline has a single state and `_study2_level` ignores the level entirely
        if any(x is not None for x in cfg.levels):
            raise ValueError(f"levels {list(cfg.levels)} on a baseline: only BDH has levels")
        levels = [None]
    elif not cfg.levels:
        levels = list(range(n_levels(model)))
    else:  # de-duplicate and sort like Study 1's explicit full_levels, and range-check first
        bad = sorted(x for x in cfg.levels if x is None or not (0 <= int(x) < n_levels(model)))
        if bad:
            raise ValueError(f"levels {bad} outside range({n_levels(model)})")
        levels = sorted({int(x) for x in cfg.levels})
    return [s for s in family_specs(shape) if not cfg.families or spec_label(s) in cfg.families], levels


def run_probes_study2(run_dir, data_dir, cfg: Study2Config, device=None) -> dict:
    """One checkpoint, one level at a time (spec section 6 memory strategy)."""
    run_dir = Path(run_dir)
    out = run_dir / "probes2"
    if (out / "done.json").exists():
        return json.loads((out / "done.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "cache"
    try:
        device = select_device(device)
        t0 = time.time()
        model, _, _ = load_checkpoint(run_dir / "ckpt.pt", device)
        is_bdh = isinstance(model, HBWMCore)
        shape = state_shape(model)
        specs, levels = _study2_selection(cfg, model, shape, is_bdh)  # fail fast: no recorder pass yet
        freqs = model.attn.freqs.reshape(-1).cpu() if is_bdh else None
        d_tr, d_va, d_te = (EpisodeData(data_dir, s) for s in ("probe_train", "probe_val", "probe_test"))
        rng = np.random.default_rng(cfg.seed)
        p_tr_all, p_va, p_te = (sample_pairs(d, rng, cfg.per_obj) for d in (d_tr, d_va, d_te))
        p_tr = stratified_subsample(p_tr_all, cfg.n_train, rng)
        n_classes = d_tr.G * d_tr.G
        chance = majority_chance(p_tr.label, p_te.label)
        ceiling = float((p_te.oracle == p_te.label).mean())
        obs_pos = tk.obs_positions(d_tr.L)
        results, summary_errors = {}, []  # summary_errors collects exploratory failures (Task 12)
        for level in levels:
            results.update(_study2_level(model, shape, freqs, (d_tr, d_va, d_te), (p_tr, p_va, p_te),
                                         specs, cfg, n_classes, device, out, cache, level, obs_pos,
                                         chance, ceiling))
            _stage_boundary(f"study2:L{level}", device)
        if cfg.bridge and not is_bdh:  # spec 5.1 cross-study bridge row; decides nothing
            p_bridge = stratified_subsample(p_tr_all, cfg.n_train_bridge, rng)
            b = _study2_level(model, shape, freqs, (d_tr, d_va, d_te), (p_bridge, p_va, p_te),
                              [FamilySpec("flat_linear")], cfg, n_classes, device, out, cache, None,
                              obs_pos, chance, ceiling, suffix="_bridge")
            for r in b.values():
                r["decides_nothing"] = True
            results.update(b)
            for label, r in b.items():
                (out / f"{label}.json").write_text(json.dumps(r, indent=2) + "\n")
    finally:  # the fp16 sigma_full memmaps are ~25 GB per level and must not survive a failure
        shutil.rmtree(cache, ignore_errors=True)
    # The H8 file the aggregator must read is the one written for the best spec across all levels.
    with_h8 = {k: v for k, v in results.items() if "h8_file" in v}
    best_overall = (max(with_h8, key=lambda k: max(with_h8[k]["val_acc"].values())) if with_h8 else None)
    summary = {"specs": sorted(k for k in results if not k.endswith("_bridge")),
               "chance": chance, "ceiling": ceiling, "n_classes": n_classes,
               "h8_file": (results[best_overall]["h8_file"] if best_overall else None),
               "h8_spec": best_overall,
               # Spec 4.5 reporting requirement: the control must be reproducible from this record.
               "randproj": {"n_out": cfg.randproj_dim, "nonzeros_per_output": cfg.randproj_density,
                            "seed": cfg.seed, "fixed_not_learned": True, "signs": [-1, 1],
                            "applied_to": "standardized_flat_sigma"},
               "shape": dataclasses.asdict(shape), "elapsed_s": round(time.time() - t0, 1),
               "probe_cfg": to_dict(cfg)}
    if summary_errors:
        summary["structure_error"] = "; ".join(summary_errors)
    (out / "done.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _study2_level(model, shape, freqs, data, pairs, specs, cfg, n_classes, device, out_dir, cache_dir,
                  level, obs_pos, chance, ceiling, suffix=""):
    d_tr, d_va, d_te = data
    p_tr, p_va, p_te = pairs
    is_bdh = isinstance(model, HBWMCore)
    flat_spec = ("sigma_full", level) if is_bdh else ("state_vec", None)
    want = [flat_spec] + ([("sigma_rownorm", level)] if any(s.input_kind == "rownorm" for s in specs) else [])
    dims = {s: feature_dim(model, s[0]) for s in want}
    memmap = (flat_spec,) if is_bdh else ()
    proj = sparse_randproj(dims[flat_spec], cfg.randproj_dim, cfg.randproj_density, cfg.seed)
    tag = "" if level is None else f"_L{level}"
    results = {}
    X = collect_many(iter_features(model, d_tr, p_tr, want, cfg.batch_eps, device), len(p_tr), dims,
                     np.float32, cache_dir, memmap)
    pos_tr = obs_pos[p_tr.t].astype(np.float32)
    pos_va = obs_pos[p_va.t].astype(np.float32)
    pos_te = obs_pos[p_te.t].astype(np.float32)
    # Spec 4.5: `mlp_randproj` projects the standardized flat state, so the flat cache's train-split
    # statistics are computed ONCE here and reused by every split. `feature_stats` streams the fp16
    # memmap in float64 row chunks, so this costs two sequential passes and no extra RAM -- and it is
    # skipped entirely when no arm in this level's set needs it.
    flat_stats = (feature_stats(X[flat_spec]) if any(s.input_kind == "randproj" for s in specs)
                  else None)
    trained, train_acc, n_in = {}, {}, {}
    for spec in specs:
        Xi = _derived_inputs(X[flat_spec], X.get(("sigma_rownorm", level)), spec.input_kind, proj,
                             flat_stats)
        n_in[spec] = Xi.shape[1]
        trained[spec] = train_family_probes(Xi, p_tr.label, n_classes, shape, spec, cfg.l2_grid,
                                            pos_tr, cfg.epochs, cfg.lr, cfg.batch, cfg.seed, device,
                                            freqs, n_in[spec])
        # Training accuracy is evidence for the preregistered degeneracy criterion (spec 7). One
        # chunked pass over Xi scores the whole (l2, restart) grid, and it must happen here, while Xi
        # is still alive: `del X` below closes the memmap.
        train_acc[spec] = evaluate_on(trained[spec], Xi, p_tr.label, pos_tr, cfg.batch, device)
        del Xi
        release_memory(device)
    del X
    _stage_boundary(f"study2{tag}:train", device)
    val = _stream_eval(model, d_va, p_va, want, flat_spec, level, trained, specs, cfg, n_classes,
                       device, proj, pos_va, flat_stats)
    best = {}
    for spec in specs:
        va = {k: accuracy(val[spec][k], p_va.label) for k in trained[spec]}
        best[spec] = max(va, key=va.get)
        results[spec_label(spec) + tag + suffix] = {
            "family": spec.name, "rank": spec.rank, "level": level, "input_kind": spec.input_kind,
            "n_features": shape.n_features, "n_train": len(p_tr), "n_val": len(p_va),
            "n_test": len(p_te), "chance": chance, "ceiling": ceiling,
            "rank_fraction": None if spec.rank is None else shape.rank_fraction(spec.rank),
            "saturated": bool(spec.rank is not None and spec.rank >= shape.saturation_rank),
            "n_params": param_count(spec.name.replace("derot_", ""), shape, n_classes, spec.rank,
                                    n_in[spec]),
            "n_input": n_in[spec],
            "n_restarts": spec.n_restarts, "best_l2": best[spec][0], "best_restart": best[spec][1],
            "val_acc": {f"{k[0]:g}/{k[1]}": v for k, v in va.items()},
            # Spec 7 degeneracy criterion: recorded as evidence here, decided in the aggregator.
            "train_acc": {f"{k[0]:g}/{k[1]}": v for k, v in train_acc[spec].items()},
        }
    del val
    _stage_boundary(f"study2{tag}:val", device)
    sel = {spec: {"best": trained[spec][best[spec]]} for spec in specs}
    del trained
    test = _stream_eval(model, d_te, p_te, want, flat_spec, level, sel, specs, cfg, n_classes, device,
                        proj, pos_te, flat_stats)
    for spec in specs:
        probs = test[spec]["best"]
        correct = probs.argmax(1) == p_te.label
        label = spec_label(spec) + tag + suffix
        r = results[label]
        r["test_acc"] = float(correct.mean())
        r["ci95"] = list(bootstrap_ci(correct, p_te.ep, cfg.n_boot))
        r["bucket_acc"] = {BUCKET_NAMES[b]: (float(correct[p_te.bucket == b].mean())
                                             if (p_te.bucket == b).any() else None)
                           for b in range(len(BUCKET_NAMES))}
        np.savez(out_dir / f"{label}_test.npz", probs=probs.astype(np.float16), label=p_te.label,
                 ep=p_te.ep, t=p_te.t, obj=p_te.obj, bucket=p_te.bucket)
        if not suffix:
            (out_dir / f"{label}.json").write_text(json.dumps(r, indent=2) + "\n")
    del test
    _stage_boundary(f"study2{tag}:test", device)
    if not suffix:  # H8 readout (spec 7) on this level's best-on-val spec
        top = max(specs, key=lambda s: max(results[spec_label(s) + tag]["val_acc"].values()))
        h3p = h3_pairs(d_te)
        if len(h3p):
            h8 = _stream_eval(model, d_te, h3p, want, flat_spec, level, {top: sel[top]}, [top], cfg,
                              n_classes, device, proj, obs_pos[h3p.t].astype(np.float32), flat_stats)
            probs = h8[top]["best"]
            rows = np.arange(len(h3p))
            # Everything the H8 statistic needs must be recomputable from this file alone, so the
            # exclusion rule and the clock rebaselining can be audited rather than trusted: per row the
            # episode, the step, whether the object was visible at that step, p(old), p(new), the
            # re-observation step, the object, and the two cell ids the probabilities were read at.
            np.savez(out_dir / f"h8{tag}.npz", p_old=probs[rows, h3p.old_cell],
                     p_new=probs[rows, h3p.new_cell], ep=h3p.ep, t=h3p.t, obj=h3p.obj,
                     old_cell=h3p.old_cell, new_cell=h3p.new_cell,
                     steps_since_reobs=h3p.steps_since_reobs,
                     reobserved_t=h3p.t - h3p.steps_since_reobs, visible_now=h3p.visible_now)
            results[spec_label(top) + tag]["h8_file"] = f"h8{tag}.npz"
            del h8
        _stage_boundary(f"study2{tag}:h8", device)
    return results


def _stream_eval(model, data, pairs, want, flat_spec, level, probes, specs, cfg, n_classes, device,
                 proj, positions, flat_stats=None):
    """One recorder pass scoring every candidate probe, with derived inputs built per batch.

    `flat_stats` is the TRAIN split's (mean, std) for the flat cache: the randproj control must see
    exactly the transform it was fitted under, never statistics refitted on val or test.
    """
    kinds = {spec: spec.input_kind for spec in specs}

    def derived():
        for idx, feats in iter_features(model, data, pairs, want, cfg.batch_eps, device):
            yield idx, {spec: _derived_inputs(feats[flat_spec], feats.get(("sigma_rownorm", level)),
                                              kinds[spec], proj, flat_stats) for spec in specs}

    return predict_proba_stream(probes, derived(), len(pairs), n_classes, device, None, positions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--data", default="data/grid9")
    ap.add_argument("--preset", default="study1", choices=sorted(PRESETS))
    ap.add_argument("--device", default=None)
    ap.add_argument("--study", default="1", choices=["1", "2"])
    ap.add_argument("--families", default="", help="comma-separated family labels; empty = all")
    ap.add_argument("--levels", default="", help="comma-separated BDH levels; empty = all")
    args = ap.parse_args()
    if args.study == "2":
        cfg = Study2Config(
            families=[f for f in args.families.split(",") if f],
            levels=[int(x) for x in args.levels.split(",") if x],
        )
        print(json.dumps(run_probes_study2(args.run_dir, args.data, cfg, args.device)))
        return
    print(json.dumps(run_probes(args.run_dir, args.data, PRESETS[args.preset], args.device)))


if __name__ == "__main__":
    main()
