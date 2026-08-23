import pytest
import torch

from hbwm.models import build_model, count_params


def test_build_bdh_and_count():
    m = build_model("bdh", {"n_layer": 2, "n_embd": 16, "n_head": 2, "mlp_internal_dim_multiplier": 8, "vocab_size": 34, "dropout": 0.0, "block_size": 128})
    assert count_params(m) == 3 * 2 * 16 * 64 + 2 * 34 * 16
    logits, loss = m(torch.randint(0, 34, (2, 10)), torch.randint(0, 34, (2, 10)))
    assert logits.shape == (2, 10, 34) and loss.ndim == 0


def test_unknown_kind():
    with pytest.raises(ValueError):
        build_model("nope", {})


def test_build_lstm_and_rwkv():
    lm = build_model("lstm", {"vocab_size": 34, "n_embd": 8, "hidden": 12, "n_layer": 2})
    r = build_model("rwkv", {"vocab_size": 34, "n_embd": 16, "n_layer": 2, "chunk_size": 5})
    x = torch.randint(0, 34, (2, 10))
    assert lm(x)[0].shape == (2, 10, 34) and r(x)[0].shape == (2, 10, 34)
