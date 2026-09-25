import torch
import torch.nn as nn

def manual_train_step(model: nn.Module, X: torch.Tensor, y: torch.Tensor, criterion: nn.Module, lr: float) -> float:
    """
    Returns the pre-update batch loss as a Python float.
    """
    prediction=model(X)
    loss=criterion(prediction,y)
    old_loss=loss.detach()
    loss.backward()
    with torch.no_grad():
      for p in model.parameters():
         p-=lr*p.grad
         p.grad=None

    return old_loss.item()
        

