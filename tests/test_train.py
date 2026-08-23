import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from hbwm.config import save_config
from hbwm.envs.dataset import EpisodeData
from hbwm.train import TrainConfig, evaluate, load_checkpoint, lr_at, run_dir, save_checkpoint, train

TINY_BDH = {"n_layer": 2, "n_embd": 16, "n_head": 2, "mlp_internal_dim_multiplier": 8, "vocab_size": 34, "dropout": 0.0, "block_size": 128}


def _cfg(tiny_data, tmp_path, **kw):
    base = dict(model="bdh", model_cfg=TINY_BDH, data_dir=tiny_data.out_dir, exp="t", run_name="bdh_tiny",
                out_root=str(tmp_path), batch_size=4, max_steps=30, lr=3e-3, warmup_steps=2, eval_every=10,
                eval_episodes=8, seed=0, device="cpu", log_every=5)
    base.update(kw)
    return TrainConfig(**base)


def test_lr_schedule():
    cfg = TrainConfig(lr=1.0, min_lr_ratio=0.1, warmup_steps=10, max_steps=110)
    assert math.isclose(lr_at(0, cfg), 0.1) and math.isclose(lr_at(9, cfg), 1.0)
    assert math.isclose(lr_at(10, cfg), 1.0)
    assert math.isclose(lr_at(110, cfg), 0.1, abs_tol=1e-9)
    assert lr_at(60, cfg) < lr_at(30, cfg) < lr_at(10, cfg)


def test_train_overfits_tiny_and_writes_artifacts(tiny_data, tmp_path):
    cfg = _cfg(tiny_data, tmp_path)
    final = train(cfg)
    rd = run_dir(cfg)
    assert (rd / "config.json").exists() and (rd / "ckpt.pt").exists() and (rd / "final.json").exists()
    lines = [json.loads(l) for l in (rd / "metrics.jsonl").read_text().splitlines()]
    train_losses = [l["train_loss"] for l in lines if "train_loss" in l]
    assert train_losses[-1] < 0.8 * train_losses[0]
    assert {"best_val_ce", "best_step", "n_params", "seconds", "lr"} <= set(final)
    assert final["n_params"] == 3 * 2 * 16 * 64 + 2 * 34 * 16


def test_evaluate_keys(tiny_data):
    from hbwm.models import build_model

    m = build_model("bdh", TINY_BDH)
    d = EpisodeData(tiny_data.out_dir, "model_val")
    out = evaluate(m, d, TrainConfig(batch_size=4, eval_episodes=8), torch.device("cpu"))
    assert set(out) == {"val_ce", "val_ce_window"} and out["val_ce"] > 0


def test_checkpoint_roundtrip(tiny_data, tmp_path):
    from hbwm.models import build_model

    cfg = _cfg(tiny_data, tmp_path)
    m = build_model("bdh", TINY_BDH)
    p = tmp_path / "c.pt"
    save_checkpoint(p, m, cfg, step=3, val_ce=1.5)
    m2, cfg2, meta = load_checkpoint(p, torch.device("cpu"))
    assert cfg2 == cfg and meta["step"] == 3
    x = torch.randint(0, 34, (1, 10))
    assert torch.equal(m.eval()(x)[0], m2.eval()(x)[0])


def test_cli_creates_run_dir(tiny_data, tmp_path):
    cfg_path = tmp_path / "bdh_tiny.json"
    save_config(_cfg(tiny_data, tmp_path, max_steps=3, eval_every=2), cfg_path)
    subprocess.run([sys.executable, "-m", "hbwm.train", "--config", str(cfg_path), "--seed", "1", "--lr", "0.001",
                    "--out-root", str(tmp_path)], check=True)
    assert (tmp_path / "t" / "bdh_tiny_lr0.001" / "seed1" / "final.json").exists()
