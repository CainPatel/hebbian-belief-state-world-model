import dataclasses

import torch
from torch import nn

from hbwm.losses import masked_ce


@dataclasses.dataclass
class LSTMConfig:
    vocab_size: int = 34
    n_embd: int = 64
    hidden: int = 350
    n_layer: int = 2
    dropout: float = 0.0


class LSTMLM(nn.Module):
    def __init__(self, cfg: LSTMConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.lstm = nn.LSTM(cfg.n_embd, cfg.hidden, num_layers=cfg.n_layer, batch_first=True,
                            dropout=cfg.dropout if cfg.n_layer > 1 else 0.0)
        self.head = nn.Linear(cfg.hidden, cfg.vocab_size)

    @property
    def state_dim(self) -> int:
        return 2 * self.cfg.n_layer * self.cfg.hidden

    def forward(self, idx, targets=None, loss_mask=None):
        out, _ = self.lstm(self.embed(idx))
        logits = self.head(out)
        loss = None if targets is None else masked_ce(logits, targets, loss_mask)
        return logits, loss

    def init_state(self, batch_size: int, device=None):
        dev = device if device is not None else self.head.weight.device
        h = torch.zeros(self.cfg.n_layer, batch_size, self.cfg.hidden, device=dev)
        return h, torch.zeros_like(h)

    def step(self, tok, state):
        out, state = self.lstm(self.embed(tok).unsqueeze(1), state)
        return self.head(out[:, 0]), state

    def state_vector(self, state):
        h, c = state
        return torch.cat([torch.cat([h[i], c[i]], dim=1) for i in range(self.cfg.n_layer)], dim=1)
