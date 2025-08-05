from models import sharpe_ratio_loss, SharpeLSTMModel, SharpeFCModel
from data.preprocessing import Data
import pandas as pd
import matplotlib.pyplot as plt
import json
import torch
from config import (
    SEED, DEVICE, HIDDEN_SIZE, FEATURE_COUNT, TIME_WINDOW, STOCK_COUNT,
    NUM_EPOCHS, BATCH_SIZE, SHARPE_WINDOW, MODEL_NAME_LSTM, MODEL_NAME_FC,
    LATEST_MODEL_PATH, LR, MOMENTUM, WEIGHT_DECAY, LOSS_FUNCTION, OPTIMIZE_TYPE,
    fix_seed, PREPROCESS_KWARGS, DF_MAG7_RAW, inspect_dataloader, NOTES_FC, NOTES_LSTM
)
from classicmethods.mpt import rolling_markowitz
import logging

logger = logging.getLogger(__name__)

def train_model(
    train_dataloader: torch.utils.data.dataloader.DataLoader,
    val_dataloader: torch.utils.data.dataloader.DataLoader,
    model_type: str = "lstm",
    lr: float = 1e-4,
    momentum: float = 0.99,
    weight_decay: float = 1e-5,
    optimize_type: str = "SGD",
    save_freq: int | None = None,
    device: torch.device = torch.device("mps")
):
    logger.info(f"Training {model_type.upper()} model...")

    # ---- Model Selection ----
    if model_type == "lstm":
        model = SharpeLSTMModel(
            num_classes=1,
            input_size=2,
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            feature_size=FEATURE_COUNT
        ).to(device)
        notes = NOTES_LSTM
        model_name = MODEL_NAME_LSTM
        plot_name = "train_val_loss_lstm"
        weight_file = "weights.pth"
        info_file = "info.json"

    elif model_type == "fc":
        model = SharpeFCModel(
            input_size=2,
            hidden_size=HIDDEN_SIZE,
            feature_size=FEATURE_COUNT
        ).to(device)
        notes = NOTES_FC
        model_name = MODEL_NAME_FC
        plot_name = "train_val_loss_fc"
        weight_file = "weights.pth"
        info_file = "info.json"

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # ---- Optimizer Selection ----
    if optimize_type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimize_type == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimize_type: {optimize_type}")

    train_losses = []
    val_losses = []

    # ---- Training Loop ----
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0.0

        for batch_data, batch_returns in train_dataloader:
            batch_data, batch_returns = batch_data.to(device), batch_returns.to(device)
            optimizer.zero_grad()

            weights = model(batch_data)
            loss = sharpe_ratio_loss(weights, batch_returns)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        # ---- Validation Loop ----
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch_data, batch_returns in val_dataloader:
                batch_data, batch_returns = batch_data.to(device), batch_returns.to(device)
                weights = model(batch_data)
                loss = sharpe_ratio_loss(weights, batch_returns)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)

        logger.info(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
        if save_freq and (epoch + 1) % save_freq == 0:
            torch.save(model.state_dict(), f"{LATEST_MODEL_PATH}/{model_type}/weights_epoch_{epoch + 1}.pth")
            logger.info(f"Saved {model_type.upper()} weights at epoch {epoch + 1}")

    # ---- Plotting ----
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label="Training Loss", color="blue")
    plt.plot(range(1, NUM_EPOCHS + 1), val_losses, label="Validation Loss", color="red")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Train & Validation Losses - {model_type.upper()}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{LATEST_MODEL_PATH}/plots/{plot_name}")
    plt.close()
    logger.info("Saved training plot.")

    # ---- Saving Model ----
    torch.save(model.state_dict(), f"{LATEST_MODEL_PATH}/{model_type}/{weight_file}")
    logger.info("Saved model weights.")

    # ---- Save Metadata ----
    metadata = {
        "model": model_name,
        "B": BATCH_SIZE,
        "R": SHARPE_WINDOW,
        "T": TIME_WINDOW,
        "S": STOCK_COUNT,
        "feature_count": FEATURE_COUNT,
        "H": HIDDEN_SIZE,
        "num_epochs": NUM_EPOCHS,
        "optimizer": optimize_type,
        "loss_function": LOSS_FUNCTION,
        "notes": notes
    }

    with open(f"{LATEST_MODEL_PATH}/{model_type}/{info_file}", 'w') as f:
        json.dump(metadata, f, indent=4)

    logger.info("Saved training metadata.")


if __name__ == "__main__":
    logger.warning("train.py executed directly")

    fix_seed(seed=SEED)
    logger.debug(f"Device: {DEVICE}, Seed: {SEED}")
    logger.debug(f"Stock count: {STOCK_COUNT}, Number of epochs: {NUM_EPOCHS}")

    stock_data = Data(DF_MAG7_RAW)
    _, test_returns = stock_data.get_test_dataframes(
        **PREPROCESS_KWARGS
    )
    train_dataloader, val_dataloader = stock_data.get_train_val_dataloaders(
        batch_size=BATCH_SIZE,
        window_size=TIME_WINDOW,
        sharpe_window=SHARPE_WINDOW,
        **PREPROCESS_KWARGS
    )
    inspect_dataloader(train_dataloader, name="Train")
    inspect_dataloader(val_dataloader, name="Val")

    train_model(
        model_type="lstm",
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        optimize_type=OPTIMIZE_TYPE,
        device=DEVICE,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        save_freq=20
        )
    
    train_model(
        model_type="fc",
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        optimize_type=OPTIMIZE_TYPE,
        device=DEVICE,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        save_freq=20
        )
    
    rolling_markowitz(
        test_returns=test_returns,
        save_path=LATEST_MODEL_PATH,
        device=DEVICE,
        learning_rate=5e-3,
        max_iter=100,
        time_window=TIME_WINDOW,
    )
