import numpy as np
import pandas as pd
import scipy.optimize as sco
# import cupy as cp
import torch
import torch.optim
import logging

logger = logging.getLogger(__name__)

def markowitz_optimization_pytorch(
    returns: torch.Tensor,
    learning_rate: float = 1e-3,
    max_iter: int = 1000,
    device: torch.device = torch.device("mps"),
):
    """
    Markowitz portfolio optimization using PyTorch with gradient descent.
    """
    # Step 1: Calculate expected returns and covariance matrix
    mean_returns = torch.mean(returns, dim=0)  # n_assets x 1
    cov_matrix = torch.cov(returns.T)  # n_assets x n_assets

    # Step 2: Initialize weights (uniform distribution, long-only portfolio)
    n_assets = returns.shape[1]
    weights = torch.ones(n_assets, dtype=torch.float32, device=device) / n_assets  # Start with equal weights
    weights.requires_grad = True  # We want to compute gradients for optimization

    # Step 3: Gradient descent to minimize portfolio variance and meet target return
    optimizer = torch.optim.SGD([weights], lr=learning_rate)  # Stochastic Gradient Descent for optimization

    for _ in range(max_iter):
        # Portfolio return and variance (risk)
        portfolio_return = torch.dot(mean_returns, weights)  # Expected return of the portfolio
        portfolio_variance = torch.matmul(weights, torch.matmul(cov_matrix, weights))

        # Portfolio loss (minimize variance)
        loss = portfolio_variance - portfolio_return

        # Add penalty to bring portfolio return closer to target_return
        # loss += torch.abs(portfolio_return - target_return) * 1000  # A large weight for the return constraint

        # Step 4: Update weights using gradient descent
        optimizer.zero_grad()  # Zero previous gradients
        loss.backward()  # Compute gradients
        optimizer.step()  # Update weights
        
        # Step 5: Normalize weights so that they sum to 1 (in-place operations)
        with torch.no_grad():
            weights.clamp_(min=0)
            weights /= weights.sum()


    return weights

def rolling_markowitz(
    test_returns: pd.DataFrame,
    save_path: str,
    device: torch.device = torch.device("mps"),
    learning_rate: float = 5e-3,
    max_iter: int = 100,
    time_window: int = 32,
):
    """
    Perform rolling Markowitz portfolio optimization on test returns. Using markowitz_optimization_pytorch function.
    Returns the portfolio value and returns for each day in the test set.
    """
    logger.info(f"Starting rolling Markowitz...")
    logger.debug(f"Iterations: {max_iter}")
    test_returns_tensor = torch.tensor(test_returns.values, dtype=torch.float32, device=device)
    
    cumulative_returns_markowitz = [1]
    markowitz_returns = []

    window_size = time_window

    for t in range(window_size, len(test_returns_tensor)):

        if t % 100 == 0:
            logger.info(f"Day {t}")

        optimal_weights = markowitz_optimization_pytorch(
            test_returns_tensor[t-window_size:t],
            learning_rate=learning_rate,
            max_iter=max_iter,
            device=device
            )

        next_day_return = torch.dot(optimal_weights, test_returns_tensor[t])

        cumulative_returns_markowitz.append(cumulative_returns_markowitz[-1] * (1 + next_day_return.item()))
        markowitz_returns.append(next_day_return.item())

    cumulative_returns_markowitz = np.array(cumulative_returns_markowitz)
    markowitz_returns = np.array(markowitz_returns)

    np.save(f"{save_path}/markowitz/cumulative_returns_markowitz.npy", cumulative_returns_markowitz)
    np.save(f"{save_path}/markowitz/markowitz_returns.npy", markowitz_returns)
    logger.info("Saved Markowitz portfolio value and returns.")
