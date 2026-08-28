import json

import numpy as np
import pytest
import torch

from hbwm.baselines.lstm import LSTMLM, LSTMConfig
from hbwm.bdh.core import HBWMConfig, HBWMCore
from hbwm.envs.dataset import EpisodeData
from hbwm.instrument.structure import measure_sigma_structure
from hbwm.probes.eligibility import sample_pairs
from hbwm.probes.probe import feature_stats
from hbwm.probes.run import Study2Config, _derived_inputs, run_probes_study2
from hbwm.probes.structured import apply_randproj, sparse_randproj
from hbwm.train import TrainConfig, save_checkpoint

TINY_BDH = HBWMConfig(n_layer=2, n_embd=16, n_head=2, mlp_internal_dim_multiplier=8, vocab_size=34,
                      dropout=0.0, block_size=128)
SMOKE = Study2Config(families=["flat_linear", "query_rank_1", "mlp_rownorm"], levels=[1],
                     l2_grid=[1e-3], epochs=1, n_restarts=1, n_train=40, per_obj=2, batch_eps=4,
                     n_boot=20, randproj_dim=8, randproj_density=4, bridge=False, structure=False)


def _never_called(*a, **kw):
    raise AssertionError("a recorder pass was started before the run's arguments were validated")


def _write_ckpt(tmp_path, model, kind, cfg_dict):
    run_dir = tmp_path / "seed0"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(run_dir / "ckpt.pt", model, TrainConfig(model=kind, model_cfg=cfg_dict), 0, 0.0)
    return run_dir


def test_study2_run_writes_one_json_per_family(tiny_data, tmp_path):
    torch.manual_seed(0)
    run_dir = _write_ckpt(tmp_path, HBWMCore(TINY_BDH), "bdh", vars(TINY_BDH))
    out = run_probes_study2(run_dir, tiny_data.out_dir, SMOKE, device="cpu")
    assert out["specs"] == ["flat_linear_L1", "mlp_rownorm_L1", "query_rank_1_L1"]
    for label in out["specs"]:
        r = json.loads((run_dir / "probes2" / f"{label}.json").read_text())
        assert 0.0 <= r["test_acc"] <= 1.0
        assert 0 < r["n_train"] <= SMOKE.n_train
        assert r["rank_fraction"] is None or 0.0 < r["rank_fraction"] <= 1.0
        assert r["n_params"] > 0 and len(r["ci95"]) == 2
        # Spec 7 needs train accuracy for every arm, at every (l2, restart) key.
        assert set(r["train_acc"]) == set(r["val_acc"])
        assert all(0.0 <= v <= 1.0 for v in r["train_acc"].values())
    assert not (run_dir / "probes2" / "cache").exists()  # try/finally deleted the memmap cache


def test_study2_run_on_a_baseline_has_no_derot_and_gains_a_bridge_row(tiny_data, tmp_path):
    torch.manual_seed(0)
    cfg = LSTMConfig(vocab_size=34, n_embd=8, hidden=6, n_layer=2)
    run_dir = _write_ckpt(tmp_path, LSTMLM(cfg), "lstm", vars(cfg))
    out = run_probes_study2(run_dir, tiny_data.out_dir,
                            Study2Config(**{**vars(SMOKE), "families": ["flat_linear", "mlp_state"],
                                            "levels": [None], "bridge": True}), device="cpu")
    assert out["specs"] == ["flat_linear", "mlp_state"]
    assert not any("derot" in s for s in out["specs"])
    bridge = json.loads((run_dir / "probes2" / "flat_linear_bridge.json").read_text())
    assert bridge["n_train"] >= json.loads((run_dir / "probes2" / "flat_linear.json").read_text())["n_train"]
    assert bridge["decides_nothing"] is True


def test_h8_readout_records_everything_needed_to_recompute_the_statistic(tiny_data, tmp_path):
    torch.manual_seed(0)
    run_dir = _write_ckpt(tmp_path, HBWMCore(TINY_BDH), "bdh", vars(TINY_BDH))
    out = run_probes_study2(run_dir, tiny_data.out_dir, SMOKE, device="cpu")
    # Spec 4.5 reporting requirement: the random projection must be reproducible from done.json.
    assert out["randproj"] == {"n_out": 8, "nonzeros_per_output": 4, "seed": 0,
                               "fixed_not_learned": True, "signs": [-1, 1],
                               "applied_to": "standardized_flat_sigma"}
    if out["h8_file"] is None:
        pytest.skip("the tiny fixture produced no moved-and-re-observed episodes")
    a = np.load(run_dir / "probes2" / out["h8_file"])
    # H8 must be auditable from this file alone: episode, step, visibility, both probabilities, the
    # re-observation step, the object, and the cells the probabilities were read at.
    assert set(a.files) >= {"p_old", "p_new", "ep", "t", "obj", "old_cell", "new_cell",
                            "steps_since_reobs", "reobserved_t", "visible_now"}
    assert (a["t"] - a["steps_since_reobs"] == a["reobserved_t"]).all()
    assert a["visible_now"].dtype == bool
    assert len(a["p_old"]) == len(a["ep"]) == len(a["visible_now"])


def test_study2_run_is_idempotent(tiny_data, tmp_path):
    torch.manual_seed(0)
    run_dir = _write_ckpt(tmp_path, HBWMCore(TINY_BDH), "bdh", vars(TINY_BDH))
    run_probes_study2(run_dir, tiny_data.out_dir, SMOKE, device="cpu")
    stamp = (run_dir / "probes2" / "done.json").stat().st_mtime_ns
    run_probes_study2(run_dir, tiny_data.out_dir, SMOKE, device="cpu")
    assert (run_dir / "probes2" / "done.json").stat().st_mtime_ns == stamp


def test_randproj_control_projects_the_standardized_flat_sigma():
    """Spec 4.5: the control is a projection of the STANDARDIZED flat sigma, not the raw row."""
    rng = np.random.default_rng(0)
    X = (rng.normal(size=(12, 32)) * np.arange(1.0, 33.0) + 5.0).astype(np.float32)
    proj = sparse_randproj(32, 6, 3, 0)
    stats = feature_stats(X)
    got = _derived_inputs(X, None, "randproj", proj, stats)
    assert np.allclose(got, apply_randproj((X - stats[0]) / stats[1], *proj), atol=1e-4)
    assert not np.allclose(got, apply_randproj(X, *proj))  # emphatically not the raw-row projection


def test_randproj_arm_uses_the_train_split_statistics_on_every_split(tiny_data, tmp_path, monkeypatch):
    """The fitted probe and both streamed passes must see ONE input distribution (spec 4.5).

    The statistics are fitted once, on the train split, and reused on val, test and the H8 pass. A
    refit inside `_stream_eval` would standardize each streamed batch by its own moments -- the exact
    failure this counts: one call per level here, many more if the statistics are ever refit.
    """
    calls = []

    def counting_feature_stats(X, *a, **kw):
        calls.append(X.shape)
        return feature_stats(X, *a, **kw)

    monkeypatch.setattr("hbwm.probes.run.feature_stats", counting_feature_stats)
    torch.manual_seed(0)
    run_dir = _write_ckpt(tmp_path, HBWMCore(TINY_BDH), "bdh", vars(TINY_BDH))
    cfg = Study2Config(**{**vars(SMOKE), "families": ["mlp_randproj"]})
    out = run_probes_study2(run_dir, tiny_data.out_dir, cfg, device="cpu")
    assert out["specs"] == ["mlp_randproj_L1"]
    r = json.loads((run_dir / "probes2" / "mlp_randproj_L1.json").read_text())
    assert r["n_input"] == cfg.randproj_dim == 8
    assert 0.0 <= r["test_acc"] <= 1.0
    # Exactly one fit, over the whole train cache, for the run's single level.
    assert len(calls) == 1, f"flat statistics fitted {len(calls)} times, not once: {calls}"
    assert calls[0] == (r["n_train"], r["n_features"])


def test_no_family_pass_is_started_for_an_unknown_family(tiny_data, tmp_path, monkeypatch):
    """A typo'd --families must not cost a ~25 GB recorder pass before it fails (spec 6)."""
    monkeypatch.setattr("hbwm.probes.run.collect_many", _never_called)
    torch.manual_seed(0)
    run_dir = _write_ckpt(tmp_path, HBWMCore(TINY_BDH), "bdh", vars(TINY_BDH))
    cfg = Study2Config(**{**vars(SMOKE), "families": ["flat_lienar", "mlp_rownorm"]})
    with pytest.raises(ValueError, match=r"unknown families \['flat_lienar'\]"):
        run_probes_study2(run_dir, tiny_data.out_dir, cfg, device="cpu")
    assert not (run_dir / "probes2" / "cache").exists()
    assert not (run_dir / "probes2" / "done.json").exists()


@pytest.mark.parametrize("levels", [[2], [-1], [0, 5]])
def test_no_pass_is_started_for_an_out_of_range_level(tiny_data, tmp_path, monkeypatch, levels):
    """-1 is the dangerous one: it indexes the LAST level while labelling every artifact `_L-1`."""
    monkeypatch.setattr("hbwm.probes.run.collect_many", _never_called)
    torch.manual_seed(0)
    run_dir = _write_ckpt(tmp_path, HBWMCore(TINY_BDH), "bdh", vars(TINY_BDH))
    cfg = Study2Config(**{**vars(SMOKE), "levels": levels})
    with pytest.raises(ValueError, match=r"outside range\(2\)"):
        run_probes_study2(run_dir, tiny_data.out_dir, cfg, device="cpu")
    assert not (run_dir / "probes2" / "cache").exists()
    assert not (run_dir / "probes2" / "done.json").exists()


def test_repeated_levels_are_deduplicated_and_sorted(tiny_data, tmp_path):
    """`--levels 1,1,0` must run each level once, not overwrite level 1 with itself."""
    torch.manual_seed(0)
    run_dir = _write_ckpt(tmp_path, HBWMCore(TINY_BDH), "bdh", vars(TINY_BDH))
    cfg = Study2Config(**{**vars(SMOKE), "families": ["flat_linear"], "levels": [1, 1, 0]})
    out = run_probes_study2(run_dir, tiny_data.out_dir, cfg, device="cpu")
    assert out["specs"] == ["flat_linear_L0", "flat_linear_L1"]


def test_measure_sigma_structure_returns_all_five_blocks(tiny_data):
    torch.manual_seed(0)
    m = HBWMCore(TINY_BDH).eval()
    d = EpisodeData(tiny_data.out_dir, "probe_train")
    pairs = sample_pairs(d, np.random.default_rng(0), per_obj=2)
    r = measure_sigma_structure(m, d, pairs, level=1, n_sample=8, seed=0, device="cpu",
                                atlas_episodes=4)
    assert set(r) == {"row_norm", "effective_rank", "activation", "write_concentration",
                      "atlas_selectivity", "n_sample", "level", "exploratory"}
    assert r["exploratory"] is True and r["n_sample"] == 8
    assert r["effective_rank"]["n_singular_values"] == TINY_BDH.n_embd


def test_measure_sigma_structure_is_invariant_to_the_episode_batch_size(tiny_data):
    """Episode batching (memory hygiene) must be numerically exact, not an approximation.

    The write-mass accumulator is per-episode independent and every reported number is an
    order-independent summary, so chunking the sampled episodes cannot move any of them.
    """

    def flat(block):
        return [v for k in sorted(block) for v in
                (sorted(block[k].values()) if isinstance(block[k], dict) else [block[k]])]

    torch.manual_seed(0)
    m = HBWMCore(TINY_BDH).eval()
    d = EpisodeData(tiny_data.out_dir, "probe_train")
    pairs = sample_pairs(d, np.random.default_rng(0), per_obj=2)
    kw = dict(level=1, n_sample=12, seed=0, device="cpu", atlas_episodes=4)
    a = measure_sigma_structure(m, d, pairs, batch_eps=1, **kw)
    b = measure_sigma_structure(m, d, pairs, batch_eps=64, **kw)
    assert a["n_sample"] == b["n_sample"]
    for name in ("row_norm", "effective_rank", "activation", "write_concentration"):
        assert flat(a[name]) == pytest.approx(flat(b[name]), rel=1e-6, abs=1e-9), name


def test_structure_success_path_writes_the_measurement_file(tiny_data, tmp_path):
    """M4's exit criterion is that `sigma_structure_L<level>.json` is PRODUCED.

    The runner call is swallowed by design, so a break there fails no test and fails no run: the
    1.5-day job would finish with the deliverable simply absent and one string in done.json. This is
    the failure-containment test minus its monkeypatch, asserting the file actually appears.
    """
    torch.manual_seed(0)
    run_dir = _write_ckpt(tmp_path, HBWMCore(TINY_BDH), "bdh", vars(TINY_BDH))
    cfg = Study2Config(**{**vars(SMOKE), "structure": True, "structure_n_sample": 8})
    out = run_probes_study2(run_dir, tiny_data.out_dir, cfg, device="cpu")
    assert not out.get("structure_error"), out.get("structure_error")
    f = run_dir / "probes2" / "sigma_structure_L1.json"
    assert f.exists()
    s = json.loads(f.read_text())
    assert set(s) == {"row_norm", "effective_rank", "activation", "write_concentration",
                      "atlas_selectivity", "n_sample", "level", "exploratory"}
    assert s["level"] == 1 and s["exploratory"] is True and s["n_sample"] == 8


def test_structure_failure_does_not_abort_the_probe_run(tiny_data, tmp_path, monkeypatch):
    import hbwm.probes.run as run_mod

    torch.manual_seed(0)
    run_dir = _write_ckpt(tmp_path, HBWMCore(TINY_BDH), "bdh", vars(TINY_BDH))
    monkeypatch.setattr(run_mod, "measure_sigma_structure",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    cfg = Study2Config(**{**vars(SMOKE), "structure": True})
    out = run_probes_study2(run_dir, tiny_data.out_dir, cfg, device="cpu")
    assert out["specs"] == ["flat_linear_L1", "mlp_rownorm_L1", "query_rank_1_L1"]
    assert "boom" in out["structure_error"]
    assert not (run_dir / "probes2" / "sigma_structure_L1.json").exists()
