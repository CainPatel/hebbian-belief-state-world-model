import numpy as np

from hbwm.bdh.core import HBWMConfig, HBWMCore

BDH_FEATURES = ("sigma_full", "sigma_rownorm", "x_sparse", "resid")
BASELINE_FEATURES = ("state_vec",)


def extract(payload: dict, name: str, level):
    """Returns [B, F] for one feature set at one level (level is ignored for baselines)."""
    if name == "sigma_full":
        s = payload["sigma"][level]  # B,nh,N,D
        return s.reshape(s.size(0), -1)
    if name == "sigma_rownorm":
        s = payload["sigma"][level]
        return s.norm(dim=-1).reshape(s.size(0), -1)  # B, nh*N
    if name == "x_sparse":
        x = payload["x_sparse"][level]  # B,nh,N
        return x.reshape(x.size(0), -1)
    if name == "resid":
        return payload["resid"][level]  # B,D
    if name == "state_vec":
        return payload["state_vec"]
    raise ValueError(name)


def n_levels(model) -> int:
    return model.hcfg.n_layer if isinstance(model, HBWMCore) else 1


def feature_dim(model, name: str) -> int:
    if isinstance(model, HBWMCore):
        c = model.hcfg
        nh, N, D = c.n_head, c.n_neurons, c.n_embd
        return {"sigma_full": nh * N * D, "sigma_rownorm": nh * N, "x_sparse": nh * N, "resid": D}[name]
    assert name == "state_vec", name
    return model.state_dim


def neuron_of_feature(cfg: HBWMConfig, name: str, f: np.ndarray) -> np.ndarray:
    """Map feature indices to neuron ids h*N + n. sigma_full is laid out (h, n, d) row-major."""
    f = np.asarray(f)
    if name == "sigma_full":
        return f // cfg.n_embd
    if name in ("sigma_rownorm", "x_sparse"):
        return f
    raise ValueError(f"no neuron mapping for {name}")
