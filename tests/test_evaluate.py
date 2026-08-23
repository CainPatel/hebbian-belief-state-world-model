import json

import numpy as np

from hbwm.matrix import GAMMA_ARMS, LRS, MODELS, run_path
from hbwm.probes.evaluate import aggregate, best_level, write_outputs

BUCKETS = ["1-4", "5-8", "9-16", "17-32", "33-64", "65+"]


def _probe_json(acc, n_features, with_h4=False, val=None):
    r = {"feature": "f", "level": 0, "n_features": n_features, "n_train": 10, "n_val": 10, "n_test": 10,
         "val_acc": {"0.001": val if val is not None else acc}, "best_l2": 0.001, "test_acc": acc, "ci95": [acc - 0.05, acc + 0.05],
         "chance": 0.05, "ceiling": 1.0, "bucket_acc": {b: max(0.0, acc - 0.05 * i) for i, b in enumerate(BUCKETS)},
         "bucket_n": {b: 10 for b in BUCKETS}}
    if with_h4:
        r["h4"] = {"ks": [16, 64, 256], "acc_by_k": {"16": acc - 0.3, "64": acc - 0.1, "256": acc - 0.02}, "acc_all": acc,
                   "neurons_by_k": {"16": 10, "64": 30, "256": 100}, "n_neurons_total": 8192}
    return r


def _level_of(spec_name):
    return int(spec_name.rsplit("_L", 1)[1]) if "_L" in spec_name else None


def _fake_run(root, stem, lr, seed, val_ce, probes: dict, h3=None, n_params=100):
    rd = run_path(root, "x", stem, lr, seed)
    (rd / "probes").mkdir(parents=True)
    (rd / "final.json").write_text(json.dumps({"best_val_ce": val_ce, "n_params": n_params, "lr": lr, "seconds": 1}))
    specs = list(probes)
    (rd / "probes" / "done.json").write_text(json.dumps({"chance": 0.05, "ceiling": 1.0, "n_classes": 81, "n_h3": 3, "specs": specs,
                                                          "best_full_spec": next((s for s in specs if s.startswith("sigma_full")), None)}))
    for name, r in probes.items():
        (rd / "probes" / f"{name}.json").write_text(json.dumps({**r, "level": _level_of(name)}))
    if h3:
        name, arrays = h3
        np.savez(rd / "probes" / f"{name}_h3.npz", **arrays)


def test_best_level_breaks_val_ties_by_lowest_level():
    # equal val accuracy, different levels; the higher level also has the higher test acc, which must not matter
    lo = {**_probe_json(0.60, 8192, val=0.70), "level": 2}
    hi = {**_probe_json(0.90, 8192, val=0.70), "level": 5}
    for probes in ({"sigma_full_L2": lo, "sigma_full_L5": hi}, {"sigma_full_L5": hi, "sigma_full_L2": lo}):
        name, r = best_level(probes, "sigma_full")
        assert name == "sigma_full_L2" and r["level"] == 2 and r["test_acc"] == 0.60


def _build_matrix(root, lstm_h4_seeds=(0, 1, 2), lstm_n_params=100):
    for stem in MODELS:  # E1 sweep artefacts so best_lr resolves to 1e-3
        for lr in LRS:
            if lr != 1e-3:
                _fake_run(root, stem, lr, 0, 2.0, {})
    h3arr = {"p_old": np.array([0.9, 0.2, 0.1]), "p_new": np.array([0.1, 0.8, 0.9]), "ep": np.array([0, 0, 1]),
             "t": np.array([5, 6, 9]), "steps_since_reobs": np.array([0, 1, 0]), "visible_now": np.array([True, False, True])}
    for seed in range(3):
        bdh_probes = {"sigma_full_L2": _probe_json(0.80 + 0.01 * seed, 524288, True), "sigma_full_L5": _probe_json(0.6, 524288, val=0.5),
                      "sigma_rownorm_L2": _probe_json(0.7, 8192), "x_sparse_L2": _probe_json(0.5, 8192), "resid_L2": _probe_json(0.4, 64)}
        _fake_run(root, "bdh_g100", 1e-3, seed, 0.5, bdh_probes, ("sigma_full_L2", h3arr))
        for arm in GAMMA_ARMS:
            _fake_run(root, arm, 1e-3, seed, 0.55, {"sigma_full_L1": _probe_json(0.7, 524288, True), "sigma_rownorm_L1": _probe_json(0.6, 8192)},
                      ("sigma_full_L1", h3arr))
        _fake_run(root, "lstm", 1e-3, seed, 0.6, {"state_vec": _probe_json(0.55, 1400, seed in lstm_h4_seeds)},
                  ("state_vec", h3arr), n_params=lstm_n_params)
        _fake_run(root, "rwkv", 1e-3, seed, 0.6, {"state_vec": _probe_json(0.5, 3520, True)}, ("state_vec", h3arr))


def test_aggregate_and_write(tmp_path):
    root = tmp_path
    _build_matrix(root)
    agg = aggregate(root, "x")
    assert agg["h1"]["supported"] and set(agg["h1"]["comparators"]) == {"x_sparse", "lstm", "rwkv"}
    assert agg["table"]["bdh_g100"]["sigma_full"]["levels"] == [2, 2, 2]
    assert agg["h2"]["bdh_g100"]["graceful"] is True and "lstm" in agg["h2"]
    assert agg["h3"]["bdh_g100"]["mean_frac_le5"] == 1.0 and agg["h3"]["bdh_g100"]["supported"]
    # exploratory H3 variant on the not-visible rows only; the headline number is untouched
    nv = agg["h3"]["bdh_g100"]["mean_frac_le5_not_visible"]
    assert isinstance(nv, float) and 0.0 <= nv <= 1.0
    assert agg["h3"]["bdh_g100"]["per_seed"][0]["exploratory_not_visible"]["n_episodes"] == 1
    assert agg["h4"]["bdh_g100"]["median_k90"] == 256 and agg["h4"]["bdh_g100"]["strong"]
    assert agg["perplexity"]["bdh_g100"]["val_ce_mean"] == 0.5
    out = tmp_path / "results"
    write_outputs(agg, out)
    assert (out / "h1.json").exists() and (out / "results.md").exists() and (out / "h2_curves.png").exists()
    md = (out / "results.md").read_text()
    assert "H1" in md and "sigma_full" in md and "lstm" in md
    assert "n_train" in md and "| 10 |" in md  # training-set size per probe
    assert "| n(1-4) |" in md and "| bdh_g100 | 10 | 10 | 10 | 10 | 10 | 10 |" in md  # H2 test-pair counts
    assert "not-visible" in md  # exploratory H3 column


def test_seed_without_h4_counts_as_infinite_k90(tmp_path):
    root = tmp_path
    _build_matrix(root, lstm_h4_seeds=(0,), lstm_n_params=None)
    agg = aggregate(root, "x")
    assert set(agg["h4"]["lstm"]["per_seed"]) == {0}  # the other two seeds have no h4 block at all
    assert agg["h4"]["lstm"]["median_k90"] is None  # median of [256, inf, inf] is inf, not 256
    assert not agg["h4"]["lstm"]["strong"] and not agg["h4"]["lstm"]["weak"]
    write_outputs(agg, tmp_path / "results")
    md = (tmp_path / "results" / "results.md").read_text()
    assert "| lstm | - | 0.001 |" in md  # a missing n_params prints "-", not "None"
    assert "| lstm | - | 1400 |" in md  # ... and so does an unreached median k90
