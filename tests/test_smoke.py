"""End-to-end on CPU: tiny data -> 3 tiny models -> probes -> decisions. Spec section 7 'smoke'."""

import json

from hbwm.probes.decisions import h1_decision
from hbwm.probes.evaluate import best_level, load_run
from hbwm.probes.run import PRESETS, run_probes
from hbwm.train import TrainConfig, run_dir, train

CFGS = {
    "bdh": {"n_layer": 2, "n_embd": 16, "n_head": 2, "mlp_internal_dim_multiplier": 8, "vocab_size": 34, "dropout": 0.0, "block_size": 128},
    "lstm": {"vocab_size": 34, "n_embd": 8, "hidden": 12, "n_layer": 2, "dropout": 0.0},
    "rwkv": {"vocab_size": 34, "n_embd": 16, "n_layer": 2, "chunk_size": 16},
}


def test_smoke_pipeline(tiny_data, tmp_path):
    accs = {}
    for kind, mcfg in CFGS.items():
        cfg = TrainConfig(model=kind, model_cfg=mcfg, data_dir=tiny_data.out_dir, exp="smoke", run_name=kind, out_root=str(tmp_path),
                          batch_size=4, max_steps=20, lr=3e-3, warmup_steps=2, eval_every=10, eval_episodes=4, seed=0, device="cpu")
        final = train(cfg)
        assert final["best_val_ce"] > 0
        rd = run_dir(cfg)
        run_probes(rd, tiny_data.out_dir, PRESETS["smoke"], device="cpu")
        run = load_run(rd)
        feat = "sigma_full" if kind == "bdh" else "state_vec"
        name, r = best_level(run["probes"], feat)
        assert r is not None and 0.0 <= r["test_acc"] <= 1.0 and "h4" in r
        accs[kind] = r["test_acc"]
    d = h1_decision([accs["bdh"]], {"lstm": [accs["lstm"]], "rwkv": [accs["rwkv"]]})
    assert set(d["comparators"]) == {"lstm", "rwkv"} and isinstance(d["supported"], bool)
