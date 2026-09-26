import torch
import torch.nn as nn
import copy

def train_with_early_stopping(model: nn.Module, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, max_epochs: int, patience: int) -> dict:
    """
    Returns train_losses and val_losses as float lists, and stopped_epoch as an int.
    """
    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    patience_counter = 0
    stopped_epoch = max_epochs

    for epoch in range(max_epochs):

        # -------------------------
        # TRAINING
        # -------------------------
        model.train()

        train_loss = 0.0

        for x, y in train_loader:

            optimizer.zero_grad()

            prediction = model(x)

            loss = criterion(prediction, y)
            train_loss += loss.item()

            loss.backward()

            optimizer.step()

          

        train_loss /= len(train_loader)

        # -------------------------
        # VALIDATION
        # -------------------------
        model.eval()

        val_loss = 0.0

        with torch.no_grad():

            for x, y in val_loader:

                prediction = model(x)

                loss = criterion(prediction, y)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Save losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # -------------------------
        # EARLY STOPPING
        # -------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            patience_counter = 0

            #best_state = copy.deepcopy(model.state_dict())

        else:

            patience_counter += 1

        if patience_counter >= patience:

            stopped_epoch = epoch + 1
            break

    # Restore best model
    

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "stopped_epoch": stopped_epoch
    }