import numpy as np
import torch

import hbwm.probes.probe as probe_mod
from hbwm.bdh.core import HBWMConfig, HBWMCore
from hbwm.envs.dataset import EpisodeData
from hbwm.instrument.features import feature_dim
from hbwm.probes.eligibility import sample_pairs
from hbwm.probes.extract import collect_many, iter_features
from hbwm.probes.probe import (
    LinearProbe,
    accuracy,
    bootstrap_ci,
    feature_stats,
    majority_chance,
    predict_proba,
    predict_proba_stream,
    train_probes_multi,
)

CFG = HBWMConfig(n_layer=2, n_embd=16, n_head=2, mlp_internal_dim_multiplier=8, vocab_size=34, dropout=0.0, block_size=128)


def test_oracle_features_give_perfect_accuracy():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, size=300)
    X = np.eye(5, dtype=np.float32)[y] * 3
    probes = train_probes_multi(X, y, 5, [1e-4, 1e-2], epochs=40, lr=1e-2, batch=64)
    assert accuracy(predict_proba(probes[1e-4], X), y) == 1.0


def test_shuffled_labels_are_at_chance():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 20)).astype(np.float32)
    y_tr, y_te = rng.integers(0, 4, size=400), rng.integers(0, 4, size=400)
    probes = train_probes_multi(X, y_tr, 4, [1e-2], epochs=10, lr=1e-2, batch=64)
    acc = accuracy(predict_proba(probes[1e-2], X), y_te)
    assert acc < majority_chance(y_tr, y_te) + 0.15


def test_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(0)
    correct = rng.random(500) < 0.7
    groups = rng.integers(0, 50, size=500)
    lo, hi = bootstrap_ci(correct, groups, n_boot=300)
    assert lo <= correct.mean() <= hi and hi - lo < 0.2


def test_iter_features_covers_every_pair_once_and_stream_matches(tiny_data):
    torch.manual_seed(0)
    m = HBWMCore(CFG).eval()
    d = EpisodeData(tiny_data.out_dir, "probe_train")
    pairs = sample_pairs(d, np.random.default_rng(0), per_obj=3)
    specs = [("sigma_rownorm", 1), ("resid", 0), ("sigma_full", 0)]
    dims = {s: feature_dim(m, s[0]) for s in specs}
    seen = []
    for idx, feats in iter_features(m, d, pairs, specs, batch_eps=4):
        seen.append(idx)
        for s in specs:
            assert feats[s].shape == (len(idx), dims[s]) and feats[s].dtype == np.float32
    seen = np.concatenate(seen)
    assert sorted(seen.tolist()) == list(range(len(pairs)))
    X = collect_many(iter_features(m, d, pairs, specs, batch_eps=4), len(pairs), dims)
    assert X[("resid", 0)].shape == (len(pairs), 16) and X[("sigma_full", 0)].dtype == np.float32
    probes = {("resid", 0): train_probes_multi(X[("resid", 0)].astype(np.float32), pairs.label, 25, [1e-3], epochs=2)}
    out = predict_proba_stream({("resid", 0): {1e-3: probes[("resid", 0)][1e-3]}},
                               iter_features(m, d, pairs, [("resid", 0)], batch_eps=4), len(pairs), 25)
    direct = predict_proba(probes[("resid", 0)][1e-3], X[("resid", 0)].astype(np.float32))
    assert np.allclose(out[("resid", 0)][1e-3], direct, atol=1e-4)
    cols = {("resid", 0): {"k4": np.array([3, 0, 9, 1])}}
    p4 = LinearProbe(4, 25)
    out4 = predict_proba_stream({("resid", 0): {"k4": p4}}, iter_features(m, d, pairs, [("resid", 0)], batch_eps=4), len(pairs), 25, columns=cols)
    assert np.allclose(out4[("resid", 0)]["k4"], predict_proba(p4, X[("resid", 0)].astype(np.float32)[:, [3, 0, 9, 1]]), atol=1e-5)


def test_topk_prefix_accuracy_is_monotone_on_synthetic():
    rng = np.random.default_rng(2)
    n, C = 600, 4
    y = rng.integers(0, C, size=n)
    signal = np.eye(C)[y] * 2.0  # the first 4 features are informative, in rank order
    X = np.concatenate([signal, rng.normal(size=(n, 16))], axis=1).astype(np.float32)
    accs = []
    for k in (1, 2, 4, 8, 20):
        p = train_probes_multi(X[:, :k], y, C, [1e-3], epochs=30, lr=1e-2, batch=64)[1e-3]
        accs.append(accuracy(predict_proba(p, X[:, :k]), y))
    assert all(b >= a - 0.03 for a, b in zip(accs, accs[1:])) and accs[2] > 0.95


def test_feature_stats_matches_numpy_and_guards_constant_columns(monkeypatch):
    rng = np.random.default_rng(3)
    X = (rng.normal(size=(37, 6)) * np.array([1.0, 5.0, 0.1, 2.0, 3.0, 100.0]) + 4.0).astype(np.float32)
    X[:, 4] = 7.5  # constant column -> std guarded to 1.0
    mean, std = feature_stats(X, chunk=8)
    assert mean.dtype == np.float32 and std.dtype == np.float32
    assert mean.shape == (6,) and std.shape == (6,)
    assert np.allclose(mean, X.mean(0), atol=1e-4)
    keep = [0, 1, 2, 3, 5]
    assert np.allclose(std[keep], X.std(0)[keep], rtol=1e-4, atol=1e-4)  # population std (ddof=0)
    assert std[4] == 1.0
    monkeypatch.setattr(probe_mod, "STATS_MAX_ELEMS", 12)  # wide-feature cap -> 2-row chunks here
    m2, s2 = feature_stats(X, chunk=4096)
    assert np.allclose(m2, mean) and np.allclose(s2, std)


def test_feature_stats_and_training_work_on_fp16_memmaps(tmp_path):
    rng = np.random.default_rng(5)
    y = rng.integers(0, 3, size=64)
    X = (np.eye(3, dtype=np.float32)[y] * 2.0 + 0.1 * rng.normal(size=(64, 3))).astype(np.float32)
    mm = np.lib.format.open_memmap(tmp_path / "f.npy", mode="w+", dtype=np.float16, shape=X.shape)
    mm[:] = X.astype(np.float16)
    mm.flush()
    mean, std = feature_stats(mm, chunk=7)  # chunked, and the rows are fp16 on disk
    assert mean.dtype == np.float32 and std.dtype == np.float32
    assert np.allclose(mean, np.asarray(mm, dtype=np.float64).mean(0), atol=1e-3)
    assert np.allclose(std, np.asarray(mm, dtype=np.float64).std(0), atol=1e-3)
    p = train_probes_multi(mm, y, 3, [1e-3], epochs=20, lr=1e-2, batch=16)[1e-3]
    assert np.allclose(p.mean.numpy(), mean) and accuracy(predict_proba(p, mm), y) == 1.0


def test_probe_standardizes_inputs_and_keeps_buffers_in_state_dict():
    torch.manual_seed(0)
    mean = np.array([1.0, -2.0, 0.5], dtype=np.float32)
    std = np.array([2.0, 0.25, 4.0], dtype=np.float32)
    p = LinearProbe(3, 4, mean=mean, std=std)
    x = torch.randn(7, 3)
    with torch.no_grad():
        expected = p.linear((x - torch.from_numpy(mean)) / torch.from_numpy(std))
        assert torch.allclose(p(x), expected)
    sd = p.state_dict()
    assert "mean" in sd and "std" in sd
    assert np.allclose(sd["mean"].numpy(), mean) and np.allclose(sd["std"].numpy(), std)
    plain = LinearProbe(3, 4)
    assert torch.equal(plain.mean, torch.zeros(3)) and torch.equal(plain.std, torch.ones(3))


def test_standardization_makes_probes_scale_invariant():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, size=300)
    X = np.eye(5, dtype=np.float32)[y] * 3
    X = (X * (10.0 ** (np.arange(5) - 2))).astype(np.float32)  # per-feature scales 1e-2 ... 1e2
    probes = train_probes_multi(X, y, 5, [1e-4, 1e-2], epochs=40, lr=1e-2, batch=64)
    assert accuracy(predict_proba(probes[1e-4], X), y) == 1.0


def test_standardization_recovers_tiny_signal_under_huge_noise():
    """The reason for the ruling: a small-scale informative feature next to large-scale uninformative
    ones is unlearnable at the fixed Study 1 recipe unless the probe standardizes its inputs."""
    rng = np.random.default_rng(0)
    n, C = 400, 4
    y = rng.integers(0, C, size=2 * n)
    X = np.concatenate([np.eye(C)[y] * 1e-3, rng.normal(size=(2 * n, 8)) * 1e2], axis=1).astype(np.float32)

    def fit(std):
        return train_probes_multi(X[:n], y[:n], C, [1e-3], epochs=20, lr=1e-3, batch=64, standardize=std)[1e-3]

    acc_on = accuracy(predict_proba(fit(True), X[n:]), y[n:])
    acc_off = accuracy(predict_proba(fit(False), X[n:]), y[n:])
    assert acc_on > 0.6 and acc_off < 0.35


def test_standardize_flag_controls_probe_buffers():
    rng = np.random.default_rng(4)
    X = (rng.normal(size=(120, 5)) * 10.0 + 3.0).astype(np.float32)
    y = rng.integers(0, 3, size=120)
    off = train_probes_multi(X, y, 3, [1e-3], epochs=2, standardize=False)[1e-3]
    assert torch.equal(off.mean, torch.zeros(5)) and torch.equal(off.std, torch.ones(5))
    on = train_probes_multi(X, y, 3, [1e-3], epochs=2)[1e-3]
    mean, std = feature_stats(X)
    assert np.allclose(on.mean.numpy(), mean) and np.allclose(on.std.numpy(), std)
