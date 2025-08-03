import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

def sharpe_ratio_loss(weights, returns, epsilon=1e-6):
    """
    weights: (B, R, S)
    returns: (B, R, S)
    """
    portfolio_returns = (weights * returns).sum(dim=2)  # (B, R)
    mean = portfolio_returns.mean(dim=1)
    std = portfolio_returns.std(dim=1) + epsilon
    sharpe = mean / std
    return -sharpe.mean()  # negative Sharpe to minimize

class SharpeLSTMModel(nn.Module):
    def __init__(self, num_classes=1, input_size=2, hidden_size=64, num_layers=1, feature_size=32):
        super(SharpeLSTMModel, self).__init__()

        self.num_classes = num_classes
        self.input_size = input_size  # 2 features (returns and prices)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.feature_size = feature_size

        # Feature projection layer (2 -> feature_size)
        self.feature_projection = nn.Linear(input_size, feature_size)

        # ReLU activation after projection layer
        self.relu = nn.ReLU()

        # LSTM layer
        self.lstm = nn.LSTM(input_size=feature_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        # Fully connected output layer (hidden_size -> 1)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        B, R, T, S, _ = x.shape

        # Project features (Returns, Prices) into higher dimensionality (B, R, T, S, F)
        x = self.feature_projection(x)  # (B, R, T, S, F)

        # Apply ReLU activation
        x = self.relu(x)  # (B, R, T, S, F)

        # Reshape to (B*R*S, T, F) for LSTM
        x = x.permute(0, 1, 3, 2, 4).reshape(B * R * S, T, self.feature_size)

        # Pass through LSTM
        lstm_out, _ = self.lstm(x)  # (B*R*S, T, H)

        # Take output of last time step
        final_out = lstm_out[:, -1, :]  # (B*R*S, H)

        # Fully connected layer
        dense_out = self.fc(final_out)  # (B*R*S, 1)

        # Reshape back to (B, R, S)
        dense_out = dense_out.view(B, R, S)  # (B, R, S)

        # Apply softmax to get weights
        weights = F.softmax(dense_out, dim=2)  # (B, R, S)

        return weights

class SharpeFCModel(nn.Module):
    def __init__(self, input_size=2, feature_size=32, hidden_size=64, dropout_rate=0.1):
        super(SharpeFCModel, self).__init__()

        self.feature_projection = nn.Linear(input_size, feature_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        # Deeper FC block: 4 layers
        self.fc1 = nn.Linear(feature_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        B, R, T, S, _ = x.shape  # (B, R, T, S, 2)

        # Project (returns, prices) → feature space
        x = self.relu(self.feature_projection(x))  # (B, R, T, S, F)

        # Pool over time (mean pooling)
        x = x.mean(dim=2)  # (B, R, S, F)

        # Feed through deeper FC block
        x = self.relu(self.fc1(x))     # (B, R, S, hidden)
        x = self.dropout(x)
        x = self.relu(self.fc2(x))     # (B, R, S, hidden)
        x = self.dropout(x)
        x = self.relu(self.fc3(x))     # (B, R, S, hidden)
        x = self.dropout(x)
        x = self.fc4(x)                # (B, R, S, 1)
        x = x.squeeze(-1)              # (B, R, S)

        # Softmax to get portfolio weights
        weights = F.softmax(x, dim=2)

        return weights

if __name__ == "__main__":
    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(module)s.%(funcName)s:%(lineno)d - %(message)s"
    )
    logger.info("models.py executed directly")