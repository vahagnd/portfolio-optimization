import numpy as np
import scipy.optimize as sco
import torch
import pandas as pd  # Assuming returns is a pandas DataFrame
import torch.optim

def markowitz_optimization(returns):
    """
    Perform Markowitz portfolio optimization to determine the optimal portfolio weights
    that minimize portfolio volatility (risk) under given constraints.

    Parameters:
        returns (pd.DataFrame): A DataFrame where each column represents the returns
                                of an asset, and each row represents a time period.

    Returns:
        np.ndarray: An array of optimal portfolio weights that minimize volatility
                    while satisfying the constraints.
    """

    # Optimization function
    def minimize_volatility(weights, mean_returns, cov_matrix):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # Compute mean & covariance
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = len(mean_returns)

    # Constraints & bounds
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_guess = num_assets * [1. / num_assets]

    # Optimize for minimum volatility
    optimized_result = sco.minimize(minimize_volatility, initial_guess,
                                    args=(mean_returns, cov_matrix),
                                    method='SLSQP', bounds=bounds, constraints=constraints)

    # Return results
    optimal_weights = optimized_result.x
    return optimal_weights



def markowitz_optimization_pytorch(returns, target_return, learning_rate=1e-3, max_iter=1000, device='cuda'):
    """
    Markowitz portfolio optimization using PyTorch with gradient descent.

    Parameters:
        returns (torch.Tensor): Matrix of asset returns (n_assets x n_days)
        target_return (float): The target portfolio return to optimize
        learning_rate (float): Learning rate for gradient descent
        max_iter (int): Maximum number of iterations for optimization
        device (str): Device to run the model on ('cuda' for GPU or 'cpu' for CPU)

    Returns:
        torch.Tensor: Optimal portfolio weights (n_assets,)
    """
    # Converting pd.DataFrame to torch.tensor
    returns = torch.tensor(returns.values, dtype=torch.float64, device=device)

    # Step 1: Calculate expected returns and covariance matrix
    mean_returns = torch.mean(returns, dim=0)  # n_assets x 1
    cov_matrix = torch.cov(returns.T)  # n_assets x n_assets

    # Step 2: Initialize weights (uniform distribution, long-only portfolio)
    n_assets = returns.shape[1]
    weights = torch.ones(n_assets, dtype=torch.float64, device=device) / n_assets  # Start with equal weights
    weights.requires_grad = True  # We want to compute gradients for optimization

    # Step 3: Gradient descent to minimize portfolio variance and meet target return
    optimizer = torch.optim.SGD([weights], lr=learning_rate)  # Stochastic Gradient Descent for optimization

    for _ in range(max_iter):
        # Portfolio return and variance (risk)
        portfolio_return = torch.dot(mean_returns, weights)  # Expected return of the portfolio
        portfolio_variance = torch.matmul(weights.T, torch.matmul(cov_matrix, weights))  # Portfolio variance (risk)

        # Portfolio loss (minimize variance)
        loss = portfolio_variance

        # Add penalty to bring portfolio return closer to target_return
        loss += torch.abs(portfolio_return - target_return) * 1000  # A large weight for the return constraint

        # Step 4: Update weights using gradient descent
        optimizer.zero_grad()  # Zero previous gradients
        loss.backward()  # Compute gradients
        optimizer.step()  # Update weights
        
        # Step 5: Normalize weights so that they sum to 1 (in-place operations)
        with torch.no_grad():
            weights.data = torch.clamp(weights.data, min=0)  # Ensure weights are non-negative (long-only portfolio)
            weights.data /= torch.sum(weights.data)  # Normalize so weights sum to 1

    return weights
