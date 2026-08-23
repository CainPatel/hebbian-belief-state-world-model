import dataclasses
import json

import numpy as np
import pytest

from hbwm.probes.eligibility import PairSet
from hbwm.probes.run import PRESETS, run_probes, stratified_subsample
from hbwm.train import TrainConfig, run_dir, train

TINY_BDH = {"n_layer": 2, "n_embd": 16, "n_head": 2, "mlp_internal_dim_multiplier": 8, "vocab_size": 34, "dropout": 0.0, "block_size": 128}
TINY_LSTM = {"vocab_size": 34, "n_embd": 8, "hidden": 12, "n_layer": 2, "dropout": 0.0}


def _train(tiny_data, tmp_path, model, mcfg, name):
    cfg = TrainConfig(model=model, model_cfg=mcfg, data_dir=tiny_data.out_dir, exp="t", run_name=name, out_root=str(tmp_path),
                      batch_size=4, max_steps=3, lr=1e-3, warmup_steps=1, eval_every=3, eval_episodes=4, seed=0, device="cpu")
    train(cfg)
    return run_dir(cfg)


def test_run_probes_bdh_and_lstm(tiny_data, tmp_path):
    rd = _train(tiny_data, tmp_path, "bdh", TINY_BDH, "bdh_tiny")
    run_probes(rd, tiny_data.out_dir, PRESETS["smoke"], device="cpu")
    out = rd / "probes"
    done = json.loads((out / "done.json").read_text())
    assert {"chance", "ceiling", "n_classes", "specs", "best_full_spec"} <= set(done) and done["n_classes"] == 25
    assert (out / "pairs_train.npz").exists() and (out / "atlas.json").exists()
    assert (out / "pairs_train_full.npz").exists()  # the rows sigma_full was actually trained on
    for lvl in (0, 1):
        r = json.loads((out / f"sigma_rownorm_L{lvl}.json").read_text())
        assert {"test_acc", "ci95", "bucket_acc", "val_acc", "best_l2", "n_features"} <= set(r) and r["n_features"] == 128
    best = done["best_full_spec"]
    r = json.loads((out / f"{best}.json").read_text())
    assert "h4" in r and r["h4"]["ks"] == [2, 4] and set(r["h4"]["acc_by_k"]) == {"2", "4"}
    assert r["h4"]["l2"] == r["best_l2"]  # top-k probes reuse the full-feature L2, not a per-k re-selection
    assert r["h4"]["neurons_by_k"]["4"] <= 4 and r["h4"]["n_neurons_total"] == 128
    assert (out / f"{best}_test.npz").exists()
    if done["n_h3"] > 0:
        z = np.load(out / f"{best}_h3.npz")
        assert z["p_old"].shape == z["p_new"].shape == (done["n_h3"],)
    assert not (out / "cache").exists()

    rd2 = _train(tiny_data, tmp_path, "lstm", TINY_LSTM, "lstm_tiny")
    run_probes(rd2, tiny_data.out_dir, PRESETS["smoke"], device="cpu")
    r = json.loads((rd2 / "probes" / "state_vec.json").read_text())
    assert r["n_features"] == 48 and "h4" in r and r["h4"]["neurons_by_k"] is None


def test_cache_removed_when_a_stage_fails(tiny_data, tmp_path, monkeypatch):
    """The fp16 sigma_full memmaps are ~25 GB per level in Study 1, so they must not survive a crash."""
    rd = _train(tiny_data, tmp_path, "bdh", TINY_BDH, "bdh_fail")

    def boom(*args, **kwargs):
        raise RuntimeError("atlas exploded")

    monkeypatch.setattr("hbwm.probes.run.build_atlas", boom)
    with pytest.raises(RuntimeError, match="atlas exploded"):
        run_probes(rd, tiny_data.out_dir, PRESETS["smoke"], device="cpu")
    # the sigma_full stage finished (so its memmaps really were written to cache/) before the atlas raised
    assert (rd / "probes" / "sigma_full_L0.json").exists() and not (rd / "probes" / "done.json").exists()
    assert not (rd / "probes" / "cache").exists()


def test_explicit_full_levels_are_deduped_sorted_and_validated(tiny_data, tmp_path):
    rd = _train(tiny_data, tmp_path, "bdh", TINY_BDH, "bdh_levels")
    pcfg = dataclasses.replace(PRESETS["smoke"], full_levels="1,1,0")
    summary = run_probes(rd, tiny_data.out_dir, pcfg, device="cpu")
    assert summary["specs"] == ["sigma_full_L0", "sigma_full_L1", "sigma_rownorm_L0", "sigma_rownorm_L1"]
    with pytest.raises(ValueError, match="full_levels"):
        run_probes(rd, tiny_data.out_dir, dataclasses.replace(pcfg, full_levels="0,2"), device="cpu")


def _synthetic_pairs(bucket_counts):
    """PairSet with the given per-bucket row counts; `ep` doubles as the original row index."""
    bucket = np.repeat(np.arange(len(bucket_counts)), bucket_counts)
    idx = np.arange(len(bucket))
    return PairSet(ep=idx, t=idx * 2, obj=idx % 3, label=idx % 25, bucket=bucket, oracle=(idx + 1) % 25,
                   sss=bucket + 1)


def test_stratified_subsample_round_robin():
    pairs = _synthetic_pairs([20, 5, 2, 1])
    sub = stratified_subsample(pairs, 12, np.random.default_rng(0))
    assert len(sub) == 12
    assert np.all(np.diff(sub.ep) > 0)  # a subset of the input rows, kept in original order
    for f in ("t", "obj", "label", "bucket", "oracle", "sss"):
        assert np.array_equal(getattr(sub, f), getattr(pairs, f)[sub.ep])  # every field subset consistently
    counts = np.bincount(sub.bucket, minlength=4)
    assert counts.tolist() == [5, 4, 2, 1]
    totals = np.bincount(pairs.bucket, minlength=4)
    # no bucket is starved while another still has leftovers: each is either exhausted or within one
    # row of the largest share
    assert all(c == t or c >= counts.max() - 1 for c, t in zip(counts, totals, strict=True))
    assert np.array_equal(stratified_subsample(pairs, 12, np.random.default_rng(0)).ep, sub.ep)  # deterministic
    other = stratified_subsample(pairs, 12, np.random.default_rng(1))
    assert np.bincount(other.bucket, minlength=4).tolist() == counts.tolist()  # balance is seed-independent


def test_stratified_subsample_keeps_everything_when_small_enough():
    pairs = _synthetic_pairs([20, 5, 2, 1])
    rng = np.random.default_rng(0)
    assert stratified_subsample(pairs, len(pairs), rng) is pairs
    assert stratified_subsample(pairs, 10_000, rng) is pairs
