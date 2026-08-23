"""Exploratory, human-readable belief readout. NOT a preregistered measurement (spec section 4.4)."""

import numpy as np
import torch

from hbwm.bdh.core import HBWMCore
from hbwm.envs import tokenizer as tk


@torch.no_grad()
def belief_map(model: HBWMCore, sigma_level, level: int, atlas: dict, obj_type: int, G: int) -> np.ndarray:
    """sigma_level: [nh, N, D] for one episode and level. Returns [G, G] indexed [y, x]:
    sum over heads of synapse(rows=cell neurons, cols=object neurons) + the transposed term."""
    sig = sigma_level.unsqueeze(0)  # [1,nh,N,D]
    cell_rows = torch.as_tensor(np.asarray(atlas["cell"])[level], device=sig.device)  # [G2, nh, m]
    obj_cols = torch.as_tensor(np.asarray(atlas["token"])[level][tk.OBJ_BASE + obj_type], device=sig.device)  # [nh, m]
    n_cells, nh, m = cell_rows.shape
    out = torch.zeros(n_cells, device=sig.device)
    for h in range(nh):
        R = cell_rows[:, h, :].reshape(-1)  # [G2*m]
        cols = obj_cols[h]  # [m]
        fwd = model.synapse(sig, level, h, R, cols)[0]  # [G2*m, m]
        out += fwd.view(n_cells, m, -1).sum(dim=(1, 2))
        bwd = model.synapse(sig, level, h, cols, R)[0]  # [m, G2*m]
        out += bwd.view(-1, n_cells, m).sum(dim=(0, 2))
    return out.view(G, G).cpu().numpy()
