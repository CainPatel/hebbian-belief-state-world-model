import json

import numpy as np

from hbwm.probes.run import PRESETS, run_probes
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
    summary = run_probes(rd, tiny_data.out_dir, PRESETS["smoke"], device="cpu")
    out = rd / "probes"
    done = json.loads((out / "done.json").read_text())
    assert {"chance", "ceiling", "n_classes", "specs", "best_full_spec"} <= set(done) and done["n_classes"] == 25
    assert (out / "pairs_train.npz").exists() and (out / "atlas.json").exists()
    for lvl in (0, 1):
        r = json.loads((out / f"sigma_rownorm_L{lvl}.json").read_text())
        assert {"test_acc", "ci95", "bucket_acc", "val_acc", "best_l2", "n_features"} <= set(r) and r["n_features"] == 128
    best = done["best_full_spec"]
    r = json.loads((out / f"{best}.json").read_text())
    assert "h4" in r and r["h4"]["ks"] == [2, 4] and set(r["h4"]["acc_by_k"]) == {"2", "4"}
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
