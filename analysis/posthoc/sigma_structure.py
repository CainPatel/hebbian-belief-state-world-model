"""Exploratory descriptive measurement: is sigma structurally sparse, and how low-rank is it?

Single checkpoint (gamma=1.0, seed0, level 3 = Study 1's selected level), N_EPISODES probe_test
episodes. Post-hoc, non-preregistered; nothing here decides anything.
"""
import time

import numpy as np
import torch

from hbwm.bdh.upstream.bdh import Attention
from hbwm.envs import tokenizer as tk
from hbwm.envs.dataset import EpisodeData
from hbwm.instrument.recorder import SigmaRecorder
from hbwm.train import load_checkpoint

# Relative to the working directory. Run this from the worktree that holds runs/ and data/, because
# the checkpoint's own cfg.data_dir ("data/grid9") is resolved relative to the cwd as well.
CKPT = "runs/study1/bdh_g100_lr0.003/seed0/ckpt.pt"
N_EPISODES = 64
LEVELS = [0, 3, 5]
EARLY_T = 5  # early observation timestep, for contrast with the final one


def pick_device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def row_stats(sigma_bh):  # [N, D] one (episode, head) slice
    norms = sigma_bh.norm(dim=-1)
    mx = norms.max().item()
    a = norms.pow(2)
    pr = (a.sum() ** 2 / a.pow(2).sum()).item()
    return {
        "mean": norms.mean().item(), "median": norms.median().item(), "max": mx,
        "frac_lt_1pct": (norms < 0.01 * mx).float().mean().item(),
        "frac_lt_10pct": (norms < 0.10 * mx).float().mean().item(),
        "pr_frac": pr / sigma_bh.shape[0],
    }


def rank_stats(sigma_bh):  # [N, D] -> SVD on CPU in double precision
    sv = torch.linalg.svdvals(sigma_bh.cpu().double())
    sq = (sv.pow(2)).float()
    frob = sq.sum()
    cum = torch.cumsum(sq, 0) / frob
    k90 = int((cum >= 0.90).nonzero()[0].item()) + 1
    k99 = int((cum >= 0.99).nonzero()[0].item()) + 1
    pr = (sq.sum() ** 2 / sq.pow(2).sum()).item()
    return {"top_sv": sq[0].sqrt().item(), "k90": k90, "k99": k99, "pr_rank": pr}


def concentration(vec):  # [N] nonneg write mass -> (top1%, top10%) mass share
    n = vec.shape[0]
    total = vec.sum()
    sv, _ = torch.sort(vec, descending=True)
    k1, k10 = max(1, round(0.01 * n)), max(1, round(0.10 * n))
    return (sv[:k1].sum() / total).item(), (sv[:k10].sum() / total).item()


def agg(vals):
    a = np.array(vals)
    return a.mean(), a.std()


def fmt(vals, k):
    m, s = agg([v[k] for v in vals])
    return f"{m:.4f}±{s:.4f}"


def main():
    dev = pick_device()
    t0 = time.time()
    model, cfg, meta = load_checkpoint(CKPT, dev)
    print(f"loaded ckpt: step={meta['step']} val_ce={meta['val_ce']:.4f} device={dev}")
    data = EpisodeData(cfg.data_dir, "probe_test")
    tok_in, _, _ = data.batch_at(np.arange(N_EPISODES), dev)
    B, Tm1 = tok_in.shape
    final_pos = Tm1 - 1
    early_pos = int(tk.obs_positions(data.L)[EARLY_T])
    print(f"B={B} T_in={Tm1} early_pos={early_pos} final_pos={final_pos} episode_len={data.L}")

    nh, N = model.hcfg.n_head, model.hcfg.n_neurons
    freqs = model.attn.freqs
    snaps = {lvl: {} for lvl in LEVELS}
    write_accum = {lvl: torch.zeros(B, nh, N, device=dev) for lvl in LEVELS}

    def fn(pos, payload):
        for lvl in LEVELS:
            xs = payload["x_sparse"][lvl]  # [B,nh,N], fresh tensor each call
            resid = payload["resid"][lvl]  # [B,D], fresh tensor each call
            phases = (float(pos) * freqs).view(1, 1, -1)
            q = Attention.rope(phases, xs)  # exact write key, matches HBWMCore.step
            write_accum[lvl] += q.pow(2) * resid.pow(2).sum(-1).view(B, 1, 1)
            if pos in (early_pos, final_pos):
                snaps[lvl][pos] = {
                    "sigma": payload["sigma"][lvl].detach().clone(),  # [B,nh,N,D]
                    "zero_frac": (xs == 0).float().mean(dim=-1).mean(dim=-1).cpu(),  # per-episode
                }

    SigmaRecorder(model).run(tok_in, None, fn)
    print(f"recorder pass over {Tm1} steps: {time.time() - t0:.1f}s\n")

    print("=== 1+2+3: row-norm sparsity, effective rank, activation sparsity (per level, snapshot) ===")
    for lvl in LEVELS:
        for pos_name, pos in (("final", final_pos), ("early", early_pos)):
            sigma = snaps[lvl][pos]["sigma"]
            rstats = [row_stats(sigma[b, h]) for b in range(B) for h in range(nh)]
            kstats = [rank_stats(sigma[b, h]) for b in range(B) for h in range(nh)]
            zf = snaps[lvl][pos]["zero_frac"]
            print(
                f"level={lvl} pos={pos_name:5s}: row_pr_frac={fmt(rstats,'pr_frac')} "
                f"frac<1%max={fmt(rstats,'frac_lt_1pct')} frac<10%max={fmt(rstats,'frac_lt_10pct')} | "
                f"svK90={fmt(kstats,'k90')}/64 svK99={fmt(kstats,'k99')}/64 "
                f"sv_pr={fmt(kstats,'pr_rank')}/64 | "
                f"x_sparse_zero_frac={zf.mean().item():.4f}±{zf.std().item():.4f}"
            )

    print("\n=== 4: write concentration (accumulated over the whole episode) ===")
    for lvl in LEVELS:
        pairs = [concentration(write_accum[lvl][b, h]) for b in range(B) for h in range(nh)]
        m1, s1 = agg([p[0] for p in pairs])
        m10, s10 = agg([p[1] for p in pairs])
        print(f"level={lvl}: top1%_write_mass={m1:.4f}±{s1:.4f} top10%_write_mass={m10:.4f}±{s10:.4f}")

    print(f"\ntotal wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
