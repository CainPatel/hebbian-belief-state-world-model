import dataclasses

import torch


@dataclasses.dataclass
class BDHState:
    """Recurrent state of HBWMCore. sigma: [L, B, nh, N, D] fp32. t: absolute index of the next token."""

    sigma: torch.Tensor
    t: int = 0

    def clone(self) -> "BDHState":
        return BDHState(sigma=self.sigma.clone(), t=self.t)
