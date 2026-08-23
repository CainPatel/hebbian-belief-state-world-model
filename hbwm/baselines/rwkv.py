"""Minimal RWKV-4-style language model (time-mix + channel-mix), pure PyTorch.
wkv_sequential is the reference; wkv_chunked (Task 17) is the training path."""

import dataclasses
import math

import torch
import torch.nn.functional as F
from torch import nn

from hbwm.losses import masked_ce


@dataclasses.dataclass
class RWKVConfig:
    vocab_size: int = 34
    n_embd: int = 176
    n_layer: int = 4
    chunk_size: int = 64


def wkv_step(w, u, k_t, v_t, aa, bb, pp):
    """One WKV step in log-space. w: [C] negative per-step log decay; u: [C] bonus. All others [B,C]."""
    ww = u + k_t
    p = torch.maximum(pp, ww)
    e1, e2 = torch.exp(pp - p), torch.exp(ww - p)
    out = (e1 * aa + e2 * v_t) / (e1 * bb + e2)
    ww = pp + w
    p = torch.maximum(ww, k_t)
    e1, e2 = torch.exp(ww - p), torch.exp(k_t - p)
    return out, e1 * aa + e2 * v_t, e1 * bb + e2, p


def wkv_sequential(w, u, k, v):
    B, T, C = k.shape
    aa = torch.zeros(B, C, dtype=k.dtype, device=k.device)
    bb = torch.zeros_like(aa)
    pp = torch.full_like(aa, float("-inf"))
    outs = []
    for t in range(T):
        o, aa, bb, pp = wkv_step(w, u, k[:, t], v[:, t], aa, bb, pp)
        outs.append(o)
    return torch.stack(outs, 1)


def wkv_chunked(w, u, k, v, chunk: int = 64):
    raise NotImplementedError("Task 17")


def _shift(x):
    """Token shift along T for [B,T,C]: x_{t-1}, zeros at t=0."""
    return F.pad(x, (0, 0, 1, -1))


def _bcast(p, x):
    return p.view((1,) * (x.dim() - 1) + (-1,))


class TimeMix(nn.Module):
    def __init__(self, C: int, layer_id: int, n_layer: int, chunk: int):
        super().__init__()
        r01 = layer_id / max(1, n_layer - 1)
        r1a0 = 1.0 - layer_id / n_layer
        h = torch.arange(C) / C
        decay_speed = -5 + 8 * (torch.arange(C) / max(1, C - 1)) ** (0.7 + 1.3 * r01)
        self.time_decay = nn.Parameter(decay_speed)  # w = -exp(time_decay)
        zigzag = (torch.arange(C) % 3 - 1) * 0.5
        self.time_first = nn.Parameter(torch.ones(C) * math.log(0.3) + zigzag)
        self.time_mix_k = nn.Parameter(h.pow(r1a0))
        self.time_mix_v = nn.Parameter(h.pow(r1a0) + 0.3 * r01)
        self.time_mix_r = nn.Parameter(0.5 * h.pow(r1a0))
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.receptance = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)
        self.chunk = chunk

    def _kvr(self, x, xx):
        mk, mv, mr = _bcast(self.time_mix_k, x), _bcast(self.time_mix_v, x), _bcast(self.time_mix_r, x)
        k = self.key(x * mk + xx * (1 - mk))
        v = self.value(x * mv + xx * (1 - mv))
        r = torch.sigmoid(self.receptance(x * mr + xx * (1 - mr)))
        return k, v, r

    def forward(self, x):
        k, v, r = self._kvr(x, _shift(x))
        wkv = wkv_sequential(-torch.exp(self.time_decay), self.time_first, k, v)  # replaced in Task 17
        return self.output(r * wkv)

    def step(self, x_t, st):
        x_prev, aa, bb, pp = st
        k, v, r = self._kvr(x_t, x_prev)
        out, aa, bb, pp = wkv_step(-torch.exp(self.time_decay), self.time_first, k, v, aa, bb, pp)
        return self.output(r * out), (x_t, aa, bb, pp)


class ChannelMix(nn.Module):
    def __init__(self, C: int, layer_id: int, n_layer: int):
        super().__init__()
        r1a0 = 1.0 - layer_id / n_layer
        h = torch.arange(C) / C
        self.time_mix_k = nn.Parameter(h.pow(r1a0))
        self.time_mix_r = nn.Parameter(h.pow(r1a0))
        self.key = nn.Linear(C, 4 * C, bias=False)
        self.value = nn.Linear(4 * C, C, bias=False)
        self.receptance = nn.Linear(C, C, bias=False)

    def _mix(self, x, xx):
        mk, mr = _bcast(self.time_mix_k, x), _bcast(self.time_mix_r, x)
        k = torch.relu(self.key(x * mk + xx * (1 - mk))) ** 2
        return torch.sigmoid(self.receptance(x * mr + xx * (1 - mr))) * self.value(k)

    def forward(self, x):
        return self._mix(x, _shift(x))

    def step(self, x_t, x_prev):
        return self._mix(x_t, x_prev), x_t


class Block(nn.Module):
    def __init__(self, C, layer_id, n_layer, chunk):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(C), nn.LayerNorm(C)
        self.tm = TimeMix(C, layer_id, n_layer, chunk)
        self.cm = ChannelMix(C, layer_id, n_layer)

    def forward(self, x):
        x = x + self.tm(self.ln1(x))
        return x + self.cm(self.ln2(x))

    def step(self, x, st):
        tm_st, cm_prev = st
        o, tm_st = self.tm.step(self.ln1(x), tm_st)
        x = x + o
        o, cm_prev = self.cm.step(self.ln2(x), cm_prev)
        return x + o, (tm_st, cm_prev)


class RWKVLM(nn.Module):
    def __init__(self, cfg: RWKVConfig):
        super().__init__()
        self.cfg = cfg
        C = cfg.n_embd
        self.embed = nn.Embedding(cfg.vocab_size, C)
        self.ln0 = nn.LayerNorm(C)
        self.blocks = nn.ModuleList([Block(C, i, cfg.n_layer, cfg.chunk_size) for i in range(cfg.n_layer)])
        self.ln_out = nn.LayerNorm(C)
        self.head = nn.Linear(C, cfg.vocab_size, bias=False)

    @property
    def state_dim(self) -> int:
        return 5 * self.cfg.n_embd * self.cfg.n_layer

    def forward(self, idx, targets=None, loss_mask=None):
        x = self.ln0(self.embed(idx))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln_out(x))
        loss = None if targets is None else masked_ce(logits, targets, loss_mask)
        return logits, loss

    def init_state(self, batch_size: int, device=None):
        dev = device if device is not None else self.head.weight.device
        C = self.cfg.n_embd
        z = lambda: torch.zeros(batch_size, C, device=dev)  # noqa: E731
        return [((z(), z(), z(), torch.full((batch_size, C), float("-inf"), device=dev)), z()) for _ in self.blocks]

    def step(self, tok, state):
        x = self.ln0(self.embed(tok))
        new = []
        for b, st in zip(self.blocks, state):
            x, st = b.step(x, st)
            new.append(st)
        return self.head(self.ln_out(x)), new

    def state_vector(self, state):
        parts = []
        for (x_prev_tm, aa, bb, pp), x_prev_cm in state:
            parts += [aa, bb, pp, x_prev_tm, x_prev_cm]
        return torch.cat(parts, dim=1)
