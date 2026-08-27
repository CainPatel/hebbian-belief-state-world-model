import torch

from hbwm.bdh.core import HBWMCore


class SigmaRecorder:
    """Drives HBWMCore.step over a batch of sequences and hands per-position payloads to a callback."""

    def __init__(self, model: HBWMCore):
        self.model = model

    @torch.no_grad()
    def run(self, tokens, positions=None, fn=None, plasticity: str = "full"):
        """payload['sigma'] (and the internals) alias the live state, which the next step() mutates in
        place: index or clone inside the callback; never store the tensor itself."""
        self.model.eval()
        B, T = tokens.shape
        want = None if positions is None else {int(p) for p in positions}
        last = T - 1 if want is None else max(want)
        state = self.model.init_state(B, tokens.device)
        for pos in range(last + 1):
            logits, state, inner = self.model.step(tokens[:, pos], state, plasticity=plasticity)
            if fn is not None and (want is None or pos in want):
                fn(pos, {"sigma": state.sigma, "x_sparse": inner["x_sparse"], "resid": inner["resid"],
                         "yKV": inner["yKV"], "logits": logits})
        return state


class StateRecorder:
    """Same contract for LSTM/RWKV baselines (models exposing init_state/step/state_vector)."""

    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def run(self, tokens, positions=None, fn=None):
        self.model.eval()
        B, T = tokens.shape
        want = None if positions is None else {int(p) for p in positions}
        last = T - 1 if want is None else max(want)
        state = self.model.init_state(B, tokens.device)
        for pos in range(last + 1):
            logits, state = self.model.step(tokens[:, pos], state)
            if fn is not None and (want is None or pos in want):
                fn(pos, {"state_vec": self.model.state_vector(state), "logits": logits})
        return state


def make_recorder(model):
    return SigmaRecorder(model) if isinstance(model, HBWMCore) else StateRecorder(model)
