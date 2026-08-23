import dataclasses

import torch
import torch.nn.functional as F
from torch import nn

from hbwm.bdh.state import BDHState
from hbwm.bdh.upstream.bdh import BDH, Attention, BDHConfig
from hbwm.losses import masked_ce


@dataclasses.dataclass
class HBWMConfig:
    n_layer: int = 6
    n_embd: int = 64
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 128
    vocab_size: int = 34
    dropout: float = 0.1
    decay_gamma: float = 1.0
    share_layers: bool = True
    block_size: int = 1164

    @property
    def n_neurons(self) -> int:
        return self.n_embd * self.mlp_internal_dim_multiplier // self.n_head

    def to_bdh(self) -> BDHConfig:
        return BDHConfig(
            n_layer=self.n_layer,
            n_embd=self.n_embd,
            dropout=self.dropout,
            n_head=self.n_head,
            mlp_internal_dim_multiplier=self.mlp_internal_dim_multiplier,
            vocab_size=self.vocab_size,
        )


def decay_mask(T: int, gamma: float, device=None) -> torch.Tensor:
    """M[t, s] = gamma^(t-1-s) for s < t, else 0. gamma == 1 gives tril(-1) of ones."""
    idx = torch.arange(T, device=device)
    diff = idx[:, None] - idx[None, :] - 1
    valid = diff >= 0
    pw = torch.pow(torch.tensor(float(gamma), device=device), diff.clamp(min=0).to(torch.float32))
    return pw * valid.to(torch.float32)


class HBWMCore(BDH):
    """Upstream BDH + gamma decay, loss mask, internals, optional per-level params, recurrent step()."""

    def __init__(self, cfg: HBWMConfig):
        super().__init__(cfg.to_bdh())
        self.hcfg = cfg
        nh, D, N, L = cfg.n_head, cfg.n_embd, cfg.n_neurons, cfg.n_layer
        if not cfg.share_layers:
            del self.encoder
            del self.encoder_v
            del self.decoder
            self.encoder_levels = nn.ParameterList(
                [nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02)) for _ in range(L)]
            )
            self.encoder_v_levels = nn.ParameterList(
                [nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02)) for _ in range(L)]
            )
            self.decoder_levels = nn.ParameterList(
                [nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02)) for _ in range(L)]
            )
        self.register_buffer("decay", decay_mask(cfg.block_size, cfg.decay_gamma), persistent=False)

    # ---- parameters per level -------------------------------------------------
    def level_params(self, level: int):
        if self.hcfg.share_layers:
            return self.encoder, self.encoder_v, self.decoder
        return self.encoder_levels[level], self.encoder_v_levels[level], self.decoder_levels[level]

    # ---- parallel path (training) --------------------------------------------
    def _attend(self, Q: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Q: [B,nh,T,N] sparse keys=queries; V: [B,1,T,D]. Returns [B,nh,T,D]."""
        T = Q.size(2)
        phases = torch.arange(T, device=Q.device, dtype=torch.float32).view(1, 1, -1, 1) * self.attn.freqs
        QR = Attention.rope(phases, Q)
        scores = (QR @ QR.mT) * self.decay[:T, :T]
        return scores @ V

    def forward(self, idx, targets=None, loss_mask=None, return_internals=False):
        C = self.hcfg
        B, T = idx.size()
        D, nh, N = C.n_embd, C.n_head, C.n_neurons
        assert T <= C.block_size, f"T={T} exceeds block_size={C.block_size}"
        x = self.ln(self.embed(idx).unsqueeze(1))  # B,1,T,D
        xs_list, resid_list = [], []
        for level in range(C.n_layer):
            enc, enc_v, dec = self.level_params(level)
            x_sparse = F.relu(x @ enc)  # B,nh,T,N
            if return_internals:
                xs_list.append(x_sparse)
                resid_list.append(x.squeeze(1))
            yKV = self.ln(self._attend(x_sparse, x))  # B,nh,T,D
            y_sparse = F.relu(yKV @ enc_v)
            xy = self.drop(x_sparse * y_sparse)
            yMLP = xy.transpose(1, 2).reshape(B, 1, T, N * nh) @ dec  # B,1,T,D
            x = self.ln(x + self.ln(yMLP))
        logits = x.view(B, T, D) @ self.lm_head
        loss = None if targets is None else masked_ce(logits, targets, loss_mask)
        if return_internals:
            return logits, loss, {"x_sparse": torch.stack(xs_list), "resid": torch.stack(resid_list)}
        return logits, loss
