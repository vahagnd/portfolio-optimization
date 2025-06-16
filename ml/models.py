import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, input_size=2, feature_size=32, hidden_size=64):
        super(SharpeFCModel, self).__init__()

        self.input_size = input_size  # 2 features: returns, prices
        self.feature_size = feature_size
        self.hidden_size = hidden_size

        # Project (returns, prices) → feature space
        self.feature_projection = nn.Linear(input_size, feature_size)
        self.relu = nn.ReLU()

        # After projection, we'll pool over time dimension
        # Then process with FC layers

        # FC block: (R * S * F) → hidden → 1 score per stock
        self.fc1 = nn.Linear(feature_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        B, R, T, S, _ = x.shape  # (B, R, T, S, 2)

        # Project features
        x = self.feature_projection(x)  # (B, R, T, S, F)
        x = self.relu(x)

        # Aggregate over time: mean pooling
        x = x.mean(dim=2)  # (B, R, S, F)

        # Flatten and feed into FC
        x = self.fc1(x)         # (B, R, S, hidden)
        x = self.relu(x)
        x = self.fc2(x)         # (B, R, S, 1)
        x = x.squeeze(-1)       # (B, R, S)

        # Apply softmax to get portfolio weights
        weights = F.softmax(x, dim=2)  # (B, R, S)

        return weights