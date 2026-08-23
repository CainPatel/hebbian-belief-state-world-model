import argparse
import dataclasses
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from hbwm.config import from_dict, load_config, save_config, to_dict
from hbwm.device import select_device
from hbwm.envs.dataset import EpisodeData
from hbwm.losses import masked_ce
from hbwm.models import build_model, count_params


@dataclasses.dataclass
class TrainConfig:
    model: str = "bdh"
    model_cfg: dict = dataclasses.field(default_factory=dict)
    data_dir: str = "data/grid9"
    exp: str = "study1"
    run_name: str = "run"
    out_root: str = "runs"
    batch_size: int = 32
    max_steps: int = 4000
    lr: float = 1e-3
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_steps: int = 200
    grad_clip: float = 1.0
    eval_every: int = 200
    eval_episodes: int = 256
    seed: int = 0
    device: str | None = None
    compile: bool = False
    log_every: int = 20


def lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    progress = min(1.0, progress)
    min_lr = cfg.lr * cfg.min_lr_ratio
    return min_lr + (cfg.lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def run_dir(cfg: TrainConfig) -> Path:
    return Path(cfg.out_root) / cfg.exp / cfg.run_name / f"seed{cfg.seed}"


@torch.no_grad()
def evaluate(model, data: EpisodeData, cfg: TrainConfig, device) -> dict:
    model.eval()
    n = min(cfg.eval_episodes, data.n)
    tot_all = tot_win = 0.0
    cnt_all = cnt_win = 0
    wmask = torch.from_numpy(data.window_mask[1:].copy()).to(device)
    for start in range(0, n, cfg.batch_size):
        idx = np.arange(start, min(n, start + cfg.batch_size))
        x, y, m = data.batch_at(idx, device)
        logits, _ = model(x)
        ce = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none").view_as(y)
        mb = m.unsqueeze(0).expand_as(y)
        wb = wmask.unsqueeze(0).expand_as(y)
        tot_all += float(ce[mb].sum())
        cnt_all += int(mb.sum())
        tot_win += float(ce[wb].sum())
        cnt_win += int(wb.sum())
    model.train()
    return {"val_ce": tot_all / cnt_all, "val_ce_window": tot_win / cnt_win}


def save_checkpoint(path, model, cfg: TrainConfig, step: int, val_ce: float) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "train_cfg": to_dict(cfg), "step": step, "val_ce": val_ce}, path)


def load_checkpoint(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = from_dict(TrainConfig, ck["train_cfg"])
    model = build_model(cfg.model, cfg.model_cfg)
    model.load_state_dict(ck["model_state"])
    model.to(device).eval()
    return model, cfg, {"step": ck["step"], "val_ce": ck["val_ce"]}


def train(cfg: TrainConfig) -> dict:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = select_device(cfg.device)
    rd = run_dir(cfg)
    rd.mkdir(parents=True, exist_ok=True)
    save_config(cfg, rd / "config.json")

    train_data = EpisodeData(cfg.data_dir, "model_train")
    val_data = EpisodeData(cfg.data_dir, "model_val")
    raw_model = build_model(cfg.model, cfg.model_cfg).to(device)
    model = raw_model
    if cfg.compile:
        model = torch.compile(model)
    n_params = count_params(raw_model)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)

    metrics_f = open(rd / "metrics.jsonl", "w")

    def log(rec):
        metrics_f.write(json.dumps(rec) + "\n")
        metrics_f.flush()

    best_val, best_step = float("inf"), -1
    t0 = time.time()
    model.train()
    for step in range(cfg.max_steps + 1):
        if step % cfg.eval_every == 0 or step == cfg.max_steps:
            ev = evaluate(model, val_data, cfg, device)
            log({"step": step, **ev, "elapsed": round(time.time() - t0, 1)})
            if ev["val_ce"] < best_val:
                best_val, best_step = ev["val_ce"], step
                save_checkpoint(rd / "ckpt.pt", raw_model, cfg, step, best_val)
        if step == cfg.max_steps:
            break
        lr = lr_at(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y, m = train_data.get_batch(rng, cfg.batch_size, device)
        _, loss = model(x, y, loss_mask=m)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        if step % cfg.log_every == 0:
            log({"step": step, "train_loss": loss.item(), "lr": lr, "elapsed": round(time.time() - t0, 1)})
    metrics_f.close()
    seconds = time.time() - t0
    final = {
        "best_val_ce": best_val,
        "best_step": best_step,
        "n_params": n_params,
        "seconds": round(seconds, 1),
        "steps_per_sec": round(cfg.max_steps / max(seconds, 1e-9), 3),
        "lr": cfg.lr,
        "model": cfg.model,
        "seed": cfg.seed,
        "device": str(device),
    }
    (rd / "final.json").write_text(json.dumps(final, indent=2) + "\n")
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, TrainConfig)
    cfg.seed = args.seed
    if args.lr is not None:
        cfg.lr = args.lr
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.out_root is not None:
        cfg.out_root = args.out_root
    if args.device is not None:
        cfg.device = args.device
    if args.data_dir is not None:
        cfg.data_dir = args.data_dir
    if args.compile:
        cfg.compile = True
    cfg.run_name = f"{Path(args.config).stem}_lr{cfg.lr:g}"
    final = train(cfg)
    print(json.dumps(final))


if __name__ == "__main__":
    main()
