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
