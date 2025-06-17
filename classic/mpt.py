import numpy as np
import pandas as pd  # Assuming returns is a pandas DataFrame
import scipy.optimize as sco
import cupy as cp
import torch
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


def markowitz_optimization_pytorch(returns, learning_rate=1e-3, max_iter=1000, device='cuda'):
    """
    Markowitz portfolio optimization using PyTorch with gradient descent.

    Parameters:
        returns (pd.Dataframe): Matrix of asset returns (n_assets x n_days)
        target_return (float): The target portfolio return to optimize
        learning_rate (float): Learning rate for gradient descent
        max_iter (int): Maximum number of iterations for optimization
        device (str): Device to run the model on ('cuda' for GPU or 'cpu' for CPU)

    Returns:
        torch.Tensor: Optimal portfolio weights (n_assets,)
    """
    returns = torch.tensor(returns.values, dtype=torch.float32, device=device)		
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
        portfolio_variance = torch.matmul(weights.T, torch.matmul(cov_matrix, weights))  # Portfolio variance (risk)

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
            weights.data = torch.clamp(weights.data, min=0)  # Ensure weights are non-negative (long-only portfolio)
            weights.data /= torch.sum(weights.data)  # Normalize so weights sum to 1

    return weights


def markowitz_closed_form(returns, target_return):
    """
    Closed-form Markowitz portfolio optimization.

    Parameters:
        mean_returns (torch.Tensor): Expected returns (n_assets,)
        cov_matrix (torch.Tensor): Covariance matrix (n_assets x n_assets)
        target_return (float): Target portfolio return

    Returns:
        torch.Tensor: Optimal portfolio weights (n_assets,)
    """
    device = returns.device

    # Mean returns and covariance matrix
    mean_returns = returns.mean(dim=1)  # (n_assets,)
    cov_matrix = torch.cov(returns)     # (n_assets x n_assets)

    ones = torch.ones_like(mean_returns, device=device)
    
    # Invert covariance matrix
    cov_inv = torch.linalg.pinv(cov_matrix)

    A = torch.dot(ones, cov_inv @ ones)
    B = torch.dot(ones, cov_inv @ mean_returns)
    C = torch.dot(mean_returns, cov_inv @ mean_returns)

    denominator = A * C - B**2
    if denominator == 0:
        raise ValueError("Singular system. Check inputs.")

    lambda_ = (A * target_return - B) / denominator
    nu = (C - B * target_return) / denominator

    weights = cov_inv @ (lambda_ * mean_returns + nu * ones)

    return weights

def markowitz_optimization_cupy(returns_df, learning_rate=1e-4, max_iter=1000, device='cuda'):
    """
    Markowitz portfolio optimization using CuPy with gradient descent on GPU.
    
    Parameters:
        returns_df (pd.DataFrame): Asset returns with shape (n_days, n_assets)
        learning_rate (float): Gradient descent learning rate
        max_iter (int): Number of iterations
        device (str): Must be 'cuda' (for PyTorch tensor output)
        
    Returns:
        torch.Tensor: Optimized portfolio weights on GPU (n_assets,)
    """
    # Convert pandas DataFrame to CuPy array on GPU
    returns_cp = cp.asarray(returns_df.values, dtype=cp.float32)  # shape (n_days, n_assets)
    
    # Calculate mean returns and covariance matrix (on GPU)
    mean_returns = cp.mean(returns_cp, axis=0)       # shape (n_assets,)
    cov_matrix = cp.cov(returns_cp.T)                # shape (n_assets, n_assets)
    
    n_assets = returns_cp.shape[1]
    weights = cp.ones(n_assets, dtype=cp.float32) / n_assets
    
    for _ in range(max_iter):
        # Gradient of portfolio variance
        grad_variance = 2 * cov_matrix.dot(weights)
        
        # Gradient of negative return (to maximize return, add minus)
        grad_return = -mean_returns
        
        # Total gradient
        grad = grad_variance + grad_return
        
        # Gradient descent step
        weights -= learning_rate * grad
        
        # Project weights: no short selling, weights >= 0, sum to 1
        weights = cp.clip(weights, 0, cp.inf)
        weights /= cp.sum(weights)

    # Move weights back to PyTorch tensor on CUDA device
    # weights_torch = torch.tensor(weights.get(), dtype=torch.float32, device=device)
    
    return weights

def differentiable_markowitz(returns, target_return, learning_rate = 1e-6):
    """
    Differentiable closed-form Markowitz optimization in PyTorch.
    
    Args:
        returns: (S, T) predicted returns, where S is number of assets, T is time
        target_return: float, the desired expected portfolio return
    
    Returns:
        weights: (S,) portfolio weights, differentiable
    """
    mu = returns.mean(dim=1)  # (S,)
    cov = returns @ returns.T / (returns.shape[1] - 1)  # (S, S)

    cov += learning_rate * torch.eye(cov.size(0), device=returns.device)  # for numerical stability

    inv_cov = torch.linalg.pinv(cov)  # (S, S)

    one = torch.ones_like(mu)  # (S,)

    A = mu @ inv_cov @ mu
    B = mu @ inv_cov @ one
    C = one @ inv_cov @ one
    D = A * C - B ** 2

    lambda1 = (C * target_return - B) / D
    lambda2 = (A - B * target_return) / D

    weights = lambda1 * (inv_cov @ mu) + lambda2 * (inv_cov @ one)  # (S,)

    weights = torch.clamp(weights, min=0)  # long-only constraint
    weights = weights / weights.sum()  # normalize

    return weights
