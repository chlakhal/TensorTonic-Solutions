import torch

def batch_norm(X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Returns a float32 tensor with the same shape as X.
    """
    mean=X.mean(dim=0)
    variance=((X-mean)**2).mean(dim=0)
    normalised=(X-mean)/torch.sqrt(variance+eps)
    return gamma * normalised + beta
