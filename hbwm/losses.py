import torch
import torch.nn.functional as F


def masked_ce(logits: torch.Tensor, targets: torch.Tensor, loss_mask: torch.Tensor | None) -> torch.Tensor:
    """Cross-entropy over positions where loss_mask is True. loss_mask: [T] or [B,T] or None."""
    if loss_mask is None:
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    m = loss_mask.to(torch.bool)
    if m.dim() == 1:
        m = m.unsqueeze(0).expand(targets.size(0), -1)
    return F.cross_entropy(logits[m], targets[m])
