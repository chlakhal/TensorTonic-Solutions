import torch

def softmax(logits: torch.Tensor) -> torch.Tensor:
    """
    Returns a float32 probability tensor with the same shape as logits.
    """
    max_val = logits.max(dim=1, keepdim=True).values
    shifted = logits - max_val
    exponential = torch.exp(shifted)
    denominator = exponential.sum(dim=1, keepdim=True)
    return exponential / denominator
