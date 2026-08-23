import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class LinearProbe(nn.Module):
    """Linear multinomial probe on per-feature standardized inputs.

    `mean`/`std` are buffers (so they move with `.to(device)` and live in `state_dict`); they default to
    the identity transform (zeros/ones). `linear.weight` is therefore the weight on *standardized*
    features, which is what H4's feature ranking compares.
    """

    def __init__(self, n_features: int, n_classes: int, mean=None, std=None):
        super().__init__()
        self.linear = nn.Linear(n_features, n_classes)
        self.register_buffer("mean", _buffer(mean, n_features, 0.0))
        self.register_buffer("std", _buffer(std, n_features, 1.0))

    def forward(self, x):
        return self.linear((x - self.mean) / self.std)


def _buffer(v, n_features: int, fill: float) -> torch.Tensor:
    """Standardization buffer of length n_features: `fill` when unset, else a private float32 copy of v."""
    if v is None:
        return torch.full((n_features,), fill, dtype=torch.float32)
    return torch.as_tensor(v, dtype=torch.float32).detach().reshape(n_features).clone()


STATS_MAX_ELEMS = 1 << 23  # cap the float64 working chunk at ~8.4M elements (~67 MB)


def feature_stats(X, chunk=4096):
    """Per-feature mean and population std as float32 arrays, accumulated in float64 over row chunks so
    X may be an in-RAM array or an on-disk fp16 memmap. Std entries < 1e-6 become 1.0, which maps
    (near-)constant features to 0 instead of amplifying their noise. The row chunk is shrunk below
    `chunk` for very wide features (sigma_full is ~5e5 columns) to bound the float64 temporaries."""
    n, F_ = X.shape
    if n == 0:
        return np.zeros(F_, dtype=np.float32), np.ones(F_, dtype=np.float32)
    rows = max(1, min(int(chunk), STATS_MAX_ELEMS // max(int(F_), 1)))
    total = np.zeros(F_, dtype=np.float64)
    for b0 in range(0, n, rows):
        total += np.asarray(X[b0 : b0 + rows], dtype=np.float64).sum(axis=0)
    mean = total / n
    sq = np.zeros(F_, dtype=np.float64)
    for b0 in range(0, n, rows):
        d = np.asarray(X[b0 : b0 + rows], dtype=np.float64) - mean
        sq += np.einsum("ij,ij->j", d, d)
    std = np.sqrt(np.maximum(sq / n, 0.0))
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def _rows(X, idx):
    idx = np.sort(idx)
    return torch.from_numpy(np.asarray(X[idx], dtype=np.float32)), idx


def train_probes_multi(X, y, n_classes, l2_grid, epochs=20, lr=1e-3, batch=512, seed=0, device="cpu",
                       standardize=True):
    """Train one linear probe per L2 value jointly (same minibatches). X: [n,F] ndarray/memmap; y: [n] ints.
    With standardize=True (Study 1 default) every probe in the grid gets the same per-feature mean/std
    taken from this training set, so features on wildly different scales stay comparable."""
    n, F_ = X.shape
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mean = std = None
    if standardize:
        mean, std = (torch.from_numpy(a) for a in feature_stats(X))
    probes = {l2: LinearProbe(F_, n_classes, mean, std).to(device) for l2 in l2_grid}
    opts = {l2: torch.optim.Adam(p.parameters(), lr=lr) for l2, p in probes.items()}
    y_t = torch.from_numpy(np.asarray(y, dtype=np.int64))
    for _ in range(epochs):
        perm = rng.permutation(n)
        for b0 in range(0, n, batch):
            xb, idx = _rows(X, perm[b0 : b0 + batch])
            xb, yb = xb.to(device), y_t[idx].to(device)
            for l2, p in probes.items():
                loss = F.cross_entropy(p(xb), yb) + l2 * p.linear.weight.pow(2).sum()
                opts[l2].zero_grad(set_to_none=True)
                loss.backward()
                opts[l2].step()
    return probes


@torch.no_grad()
def predict_proba(probe, X, batch=4096, device="cpu"):
    probe = probe.to(device).eval()
    out = []
    for b0 in range(0, X.shape[0], batch):
        xb = torch.from_numpy(np.asarray(X[b0 : b0 + batch], dtype=np.float32)).to(device)
        out.append(torch.softmax(probe(xb), dim=-1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, probe.linear.out_features))


@torch.no_grad()
def predict_proba_stream(probes, it, n, n_classes, device="cpu", columns=None):
    """probes: {spec: {key: LinearProbe}}. columns: optional {spec: {key: feature-index array}} to slice rows
    (applied to the raw streamed features *before* the probe, which standardizes its own subset)."""
    out = {s: {k: np.zeros((n, n_classes), dtype=np.float32) for k in d} for s, d in probes.items()}
    for idx, feats in it:
        for s, d in probes.items():
            xb_full = torch.from_numpy(feats[s]).to(device)
            for k, p in d.items():
                xb = xb_full
                if columns is not None and s in columns and k in columns[s]:
                    xb = xb_full[:, torch.as_tensor(columns[s][k], device=device)]
                out[s][k][idx] = torch.softmax(p.to(device).eval()(xb), dim=-1).cpu().numpy()
    return out


def accuracy(probs, y) -> float:
    return float((probs.argmax(1) == np.asarray(y)).mean()) if len(y) else float("nan")


def bootstrap_ci(correct, groups, n_boot=1000, seed=0, alpha=0.05):
    correct, groups = np.asarray(correct, dtype=np.float64), np.asarray(groups)
    ug, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=correct, minlength=len(ug))
    cnts = np.bincount(inv, minlength=len(ug)).astype(np.float64)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(ug), size=len(ug))
        means.append(sums[pick].sum() / cnts[pick].sum())
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def majority_chance(y_train, y_test) -> float:
    maj = np.bincount(np.asarray(y_train)).argmax()
    return float((np.asarray(y_test) == maj).mean())
