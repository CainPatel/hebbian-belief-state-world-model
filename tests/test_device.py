import torch
from hbwm.device import select_device


def test_prefer_overrides():
    assert select_device("cpu") == torch.device("cpu")


def test_auto_returns_known_type():
    assert select_device().type in {"mps", "cuda", "cpu"}
