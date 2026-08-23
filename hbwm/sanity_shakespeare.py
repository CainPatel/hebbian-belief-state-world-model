"""E0 sanity: train HBWMCore (gamma=1, byte-level) on tiny-Shakespeare on the local device.
Not a unit test: a loss curve that falls monotonically and a non-degenerate sample are the check."""

import argparse
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch

from hbwm.bdh.core import HBWMConfig, HBWMCore
from hbwm.device import select_device

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="runs/e0_shakespeare")
    args = ap.parse_args()

    data_path = Path("data/shakespeare/input.txt")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    if not data_path.exists():
        urllib.request.urlretrieve(URL, data_path)
    data = np.frombuffer(data_path.read_bytes(), dtype=np.uint8)
    n_train = int(0.9 * len(data))
    train, val = data[:n_train], data[n_train:]

    device = select_device(args.device)
    torch.manual_seed(1337)
    cfg = HBWMConfig(n_layer=6, n_embd=128, n_head=4, mlp_internal_dim_multiplier=128,
                     vocab_size=256, dropout=0.1, decay_gamma=1.0, block_size=args.block)
    model = HBWMCore(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)

    def batch(arr):
        ix = np.random.randint(0, len(arr) - args.block - 1, size=args.batch)
        x = torch.from_numpy(np.stack([arr[i:i + args.block].astype(np.int64) for i in ix]))
        y = torch.from_numpy(np.stack([arr[i + 1:i + 1 + args.block].astype(np.int64) for i in ix]))
        return x.to(device), y.to(device)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = []
    t0 = time.time()
    for step in range(args.steps + 1):
        if step % 100 == 0:
            model.eval()
            with torch.no_grad():
                vx, vy = batch(val)
                _, vloss = model(vx, vy)
            model.train()
            rec = {"step": step, "val_loss": float(vloss), "elapsed_s": round(time.time() - t0, 1)}
            log.append(rec)
            print(json.dumps(rec), flush=True)
        if step == args.steps:
            break
        x, y = batch(train)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    prompt = torch.tensor(list(b"To be or "), dtype=torch.long, device=device).unsqueeze(0)
    sample = model.generate(prompt, max_new_tokens=200, top_k=3)
    text = bytes(sample[0].to("cpu").to(torch.uint8).tolist()).decode(errors="backslashreplace")
    (out / "log.json").write_text(json.dumps({"n_params": n_params, "device": str(device), "log": log, "sample": text}, indent=2))
    print(f"n_params={n_params} device={device}\n{text}")


if __name__ == "__main__":
    main()
