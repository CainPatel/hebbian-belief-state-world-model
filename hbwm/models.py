from torch import nn

from hbwm.bdh.core import HBWMConfig, HBWMCore
from hbwm.config import from_dict


def build_model(kind: str, model_cfg: dict) -> nn.Module:
    if kind == "bdh":
        return HBWMCore(from_dict(HBWMConfig, model_cfg))
    if kind == "lstm":
        from hbwm.baselines.lstm import LSTMConfig, LSTMLM

        return LSTMLM(from_dict(LSTMConfig, model_cfg))
    if kind == "rwkv":
        from hbwm.baselines.rwkv import RWKVConfig, RWKVLM

        return RWKVLM(from_dict(RWKVConfig, model_cfg))
    raise ValueError(f"unknown model kind {kind!r}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
