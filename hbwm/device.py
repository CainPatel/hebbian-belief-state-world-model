import gc

import torch


def select_device(prefer: str | None = None) -> torch.device:
    """mps > cuda > cpu unless `prefer` names a device explicitly."""
    if prefer is not None:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def release_memory(device=None) -> None:
    """Collect Python garbage and hand the MPS allocator's cached blocks back to the system.

    Numerically inert: it only releases allocations nothing references any more, so it is safe to call
    between (never inside) recorder passes, whose payloads alias live state. `device` narrows the work:
    a non-MPS device skips the MPS call entirely, so CPU-only runs on an Apple machine pay nothing
    beyond the `gc.collect()`. Lives here, in the layer both `hbwm.probes` and `hbwm.instrument`
    already depend on, so neither package has to import the other for it.
    """
    gc.collect()
    if device is not None and torch.device(device).type != "mps":
        return
    mps = getattr(torch, "mps", None)
    if torch.backends.mps.is_available() and hasattr(mps, "empty_cache"):
        mps.empty_cache()
