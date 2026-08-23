import hashlib
from pathlib import Path

import torch

from hbwm.bdh.upstream.bdh import BDH, BDHConfig


def test_upstream_forward_shape():
    cfg = BDHConfig(n_layer=2, n_embd=16, n_head=2, mlp_internal_dim_multiplier=8, vocab_size=10, dropout=0.0)
    m = BDH(cfg).eval()
    idx = torch.randint(0, 10, (3, 7))
    logits, loss = m(idx, idx)
    assert logits.shape == (3, 7, 10) and loss.ndim == 0


def test_upstream_file_unmodified():
    p = Path(__file__).resolve().parents[1] / "hbwm" / "bdh" / "upstream" / "bdh.py"
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    recorded = (p.parent / "UPSTREAM.md").read_text()
    assert digest in recorded, "bdh.py hash is not the one recorded in UPSTREAM.md"
