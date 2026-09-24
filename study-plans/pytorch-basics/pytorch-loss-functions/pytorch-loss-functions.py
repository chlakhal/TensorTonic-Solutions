import torch

def compute_loss(pred: torch.Tensor, target: torch.Tensor, method: str, delta: float = 1.0) -> float:
    """
    Returns the mean loss as a Python float.
    """
    if method == "mse":
        return ((pred - target) ** 2).mean().item()
    if method == "cross_entropy":
        shifted = pred - pred.max(dim=1, keepdim=True).values
        log_partition = shifted.exp().sum(dim=1).log()
        target_logits = shifted[torch.arange(pred.shape[0]), target]
        return (log_partition - target_logits).mean().item()
    difference = (pred - target).abs()
    losses = torch.where(difference <= delta, 0.5 * difference ** 2,
                         delta * (difference - 0.5 * delta))
    return losses.mean().item()