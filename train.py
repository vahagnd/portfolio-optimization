from models import sharpe_ratio_loss, SharpeLSTMModel, SharpeFCModel
from dataloaders import get_dataloaders
from data.preprocessing import Data
import pandas as pd
import matplotlib.pyplot as plt
import json
import torch
from config import *
import logging

logger = logging.getLogger(__name__)

train_dataloader, val_dataloader = get_dataloaders(DF_MAG7_RAW, years=YEARS, prices=PRICES, normalize=True, scaler=SCALER)

logger.info(f"Stock count: {STOCK_COUNT}")

def train_lstm(lr: float = 1e-4, momentum: float = 0.99, weight_decay: float = 1e-5, optimize_type: str = "SGD"):
    logger.info("Training LSTM model...")
    model = SharpeLSTMModel(num_classes=1, input_size=2, hidden_size=HIDDEN_SIZE, num_layers=1, feature_size=FEATURE_COUNT).to(DEVICE)
    
    if optimize_type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimize_type == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        pass

    train_losses = []
    val_losses = []

    for epoch in range(NUM_EPOCHS):
        model.train()  # Set the model to training mode
        total_train_loss = 0.0
        for batch_data, batch_returns in train_dataloader:
            batch_data, batch_returns = batch_data.to(DEVICE), batch_returns.to(DEVICE)

            optimizer.zero_grad()  # Reset gradients

            # Forward pass
            weights = model(batch_data)

            # Compute training loss (Negative Sharpe ratio)
            train_loss = sharpe_ratio_loss(weights, batch_returns)
            total_train_loss += train_loss.item()

            # Backward pass
            train_loss.backward()

            # Update parameters
            optimizer.step()

        avg_train_loss = total_train_loss / len(train_dataloader)  # Average training loss for the epoch
        train_losses.append(avg_train_loss) # To plot after

        # Validation phase (after each epoch)
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # No gradient calculation needed for validation
            total_val_loss = 0.0
            for batch_data, batch_returns in val_dataloader:
                batch_data, batch_returns = batch_data.to(DEVICE), batch_returns.to(DEVICE)

                # Forward pass (no gradients needed)
                weights = model(batch_data)

                # Compute validation loss (Negative Sharpe ratio)
                val_loss = sharpe_ratio_loss(weights, batch_returns)
                total_val_loss += val_loss.item()

            avg_val_loss = total_val_loss / len(val_dataloader)  # Average validation loss for the epoch
            val_losses.append(avg_val_loss)

        # Print training and validation loss for the current epoch
        logger.info(f"Epoch {epoch+1}: Training Loss = {avg_train_loss:.6f}, Validation Loss = {avg_val_loss:.6f}")

    # Plotting train losses per epoch LSTM
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label="Training Loss", color='blue')
    plt.plot(range(1, NUM_EPOCHS + 1), val_losses, label="Validation Loss", color='red')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Average Training and Validation Losses per Epoch, LSTM")
    plt.legend()

    # Saving plot
    logger.info("Saving loss plot...")
    plt.savefig(f"{LATEST_MODEL_PATH}/train_val_loss")

    plt.show()

    # Saving model
    logger.info("Saving model...")
    torch.save(model.state_dict(), f"{LATEST_MODEL_PATH}/weights.pth")
    metadata = {
        "model": MODEL_NAME_FC,
        "B": BATCH_SIZE,
        "R": SHARPE_WINDOW,
        "T": TIME_WINDOW,
        "S": STOCK_COUNT,
        "feature_count": FEATURE_COUNT,
        "H": HIDDEN_SIZE,
        "num_epochs": NUM_EPOCHS,
        "optimizer": OPTIMIZE_TYPE,
        "loss_function": LOSS_FUNCTION,
        "notes": "fp-relu-lstm-fc-sm"
    }

    with open(f"{LATEST_MODEL_PATH}/info.json", 'w') as f:
        json.dump(metadata, f, indent=4)


def train_fc(lr: float = 1e-4, momentum: float = 0.99, weight_decay: float = 1e-5, optimize_type: str = "SGD"):
    logger.info("Training FC model...")
    model_fc = SharpeFCModel(input_size=2, hidden_size=HIDDEN_SIZE, feature_size=FEATURE_COUNT).to(DEVICE)

    if optimize_type == "Adam":
        optimizer_fc = torch.optim.Adam(model_fc.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimize_type == "SGD":
        optimizer_fc = torch.optim.SGD(model_fc.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        pass

    train_losses_fc = []
    val_losses_fc = []

    for epoch in range(NUM_EPOCHS):
        model_fc.train()  # Set model_fc to training mode
        total_train_loss = 0.0
        for batch_data, batch_returns in train_dataloader:
            batch_data, batch_returns = batch_data.to(DEVICE), batch_returns.to(DEVICE)

            optimizer_fc.zero_grad()  # Reset gradients for model_fc

            # Forward pass
            weights = model_fc(batch_data)

            # Compute training loss (Negative Sharpe ratio)
            train_loss = sharpe_ratio_loss(weights, batch_returns)
            total_train_loss += train_loss.item()

            # Backward pass
            train_loss.backward()

            # Update parameters
            optimizer_fc.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        train_losses_fc.append(avg_train_loss)

        # Validation phase
        model_fc.eval()
        with torch.no_grad():
            total_val_loss = 0.0
            for batch_data, batch_returns in val_dataloader:
                batch_data, batch_returns = batch_data.to(DEVICE), batch_returns.to(DEVICE)

                # Forward pass
                weights = model_fc(batch_data)

                # Compute validation loss (Negative Sharpe ratio)
                val_loss = sharpe_ratio_loss(weights, batch_returns)
                total_val_loss += val_loss.item()

            avg_val_loss = total_val_loss / len(val_dataloader)
            val_losses_fc.append(avg_val_loss)

        # Print losses
        logger.info(f"Epoch {epoch+1}: Training Loss = {avg_train_loss:.6f}, Validation Loss = {avg_val_loss:.6f}")

    # Plotting train losses per epoch FC
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, NUM_EPOCHS + 1), train_losses_fc, label="Training Loss", color='blue')
    plt.plot(range(1, NUM_EPOCHS + 1), val_losses_fc, label="Validation Loss", color='red')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Average Training and Validation Losses per Epoch, FC")
    plt.legend()

    # # Saving plot
    logger.info("Saving loss plot...")
    plt.savefig(f"{LATEST_MODEL_PATH}/train_val_loss_fc")
    
    # Saving model
    logger.info("Saving model...")
    torch.save(model_fc.state_dict(), f"{LATEST_MODEL_PATH}/weights_fc.pth")
    metadata = {
        "model": MODEL_NAME_FC,
        "B": BATCH_SIZE,
        "R": SHARPE_WINDOW,
        "T": TIME_WINDOW,
        "S": STOCK_COUNT,
        "feature_count": FEATURE_COUNT,
        "H": HIDDEN_SIZE,
        "num_epochs": NUM_EPOCHS,
        "optimizer": OPTIMIZE_TYPE,
        "loss_function": LOSS_FUNCTION,
        "notes": "fp-relu-mp-fc-relu-fc-relu-fc-relu-fc-sm"
    }

    with open(f"{LATEST_MODEL_PATH}/info_fc.json", 'w') as f:
        json.dump(metadata, f, indent=4)


if __name__ == "__main__":
    logger.info("train.py executed directly")

    fix_seed(seed=SEED)
    logger.info(f"Device: {DEVICE}, Seed: {SEED}")
    
    train_lstm(lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, optimize_type=OPTIMIZE_TYPE)
    train_fc(lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, optimize_type=OPTIMIZE_TYPE)