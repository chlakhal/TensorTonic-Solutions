import torch
import torch.nn as nn

def train_with_scheduler(model: nn.Module, dataloader: torch.utils.data.DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.StepLR, num_epochs: int) -> dict:
    """
    Returns losses and lrs as lists of Python floats in a dictionary.
    """
    losses=[]
    lrs=[]
    for epoch in range(num_epochs):
        # LR used during THIS epoch
         lrs.append(float(optimizer.param_groups[0]["lr"]))

         batch_losses = []

         for X, y in dataloader:

            optimizer.zero_grad()

            prediction = model(X)
            loss = criterion(prediction, y)

            batch_losses.append(loss.item())

            loss.backward()
            optimizer.step()

         losses.append(sum(batch_losses) / len(batch_losses))

         scheduler.step()
    return {
        "losses": losses,
         "lrs": lrs
       
    }
