from typing import List, Tuple
import numpy as np
from ml.domain.models import IterationStep


def gradient_descent(
    X: List[float],
    y: List[float],
    alpha: float = 0.01,
    iterations: int = 100
) -> Tuple[List[float], List[IterationStep], float, float]:
    """
    Gradiente descendente para regresion lineal simple.

    Formulas:
      h(x) = theta_0 + theta_1 * x
      J(theta) = (1/2m) * sum(h(x_i) - y_i)^2
      dJ/dtheta_0 = (1/m) * sum(h(x_i) - y_i)
      dJ/dtheta_1 = (1/m) * sum(h(x_i) - y_i) * x_i
      theta_j := theta_j - alpha * dJ/dtheta_j
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    m = len(y)

    x_mean, x_std = X.mean(), X.std()
    if x_std == 0:
        x_std = 1
    X_norm = (X - x_mean) / x_std

    theta = np.zeros(2)
    history = []

    for i in range(iterations):
        h = theta[0] + theta[1] * X_norm
        error = h - y

        grad_0 = (1 / m) * np.sum(error)
        grad_1 = (1 / m) * np.sum(error * X_norm)

        theta[0] -= alpha * grad_0
        theta[1] -= alpha * grad_1

        cost = (1 / (2 * m)) * np.sum(error ** 2)

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