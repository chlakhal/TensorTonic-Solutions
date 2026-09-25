import torch
import torch.nn as nn

def train_epoch(model: nn.Module, dataloader: torch.utils.data.DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer) -> float:
    """
    Returns the mean batch loss as a Python float.
    """
    model.train()
    total_loss=0.0
    for x,y in dataloader:
        optimizer.zero_grad()
        prediction=model(x)
        loss=criterion(prediction,y)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    return total_loss/len(dataloader)
