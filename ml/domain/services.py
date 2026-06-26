from typing import List, Tuple
from ml.domain.models import IterationStep


def gradient_descent(X: List[float], y: List[float], alpha: float = 0.01, iterations: int = 100) -> Tuple[List[float], List[IterationStep]]:
    """
    Gradiente descendente para regresión lineal simple.
    
    Fórmulas:
      h(x) = θ₀ + θ₁·x
      J(θ) = (1/2m) · Σ(h(xᵢ) - yᵢ)²
      ∂J/∂θ₀ = (1/m) · Σ(h(xᵢ) - yᵢ)
      ∂J/∂θ₁ = (1/m) · Σ(h(xᵢ) - yᵢ)·xᵢ
      θⱼ := θⱼ - α · ∂J/∂θⱼ
    """
    import numpy as np
    
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    m = len(y)
    
    x_mean, x_std = X.mean(), X.std()
    X_norm = (X - x_mean) / x_std
    
    theta = np.zeros(2)
    history = []
    
    for i in range(iterations):
        h = theta[0] + theta[1] * X_norm
        error = h - y
        
        grad_0 = (1/m) * np.sum(error)
        grad_1 = (1/m) * np.sum(error * X_norm)
        
        theta[0] -= alpha * grad_0
        theta[1] -= alpha * grad_1
        
        cost = (1/(2*m)) * np.sum(error**2)
        
        predictions = theta[0] + theta[1] * X_norm
        
        theta_0_real = float(theta[0] - theta[1] * (x_mean / x_std))
        theta_1_real = float(theta[1] / x_std)
        
        history.append(IterationStep(
            iteration=i + 1,
            theta_0=float(theta[0]),
            theta_1=float(theta[1]),
            theta_0_real=theta_0_real,
            theta_1_real=theta_1_real,
            cost=float(cost),
            gradient_0=float(grad_0),
            gradient_1=float(grad_1),
            predictions=predictions.tolist()
        ))
    
    theta_real = [
        float(theta[0] - theta[1] * (x_mean / x_std)),
        float(theta[1] / x_std)
    ]
    
    return theta_real, history, float(x_mean), float(x_std)