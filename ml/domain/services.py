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


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def logistic_gradient_descent(
    X: List[float],
    y: List[float],
    alpha: float = 0.01,
    iterations: int = 100
) -> Tuple[List[float], List[IterationStep], float, float]:
    """
    Regresion logistica con gradiente descendente.

    Formulas:
      g(z) = 1 / (1 + e^-z)
      h(x) = g(theta_0 + theta_1 * x)
      J(theta) = -(1/m) * sum(y * log(h) + (1-y) * log(1-h))
      dJ/dtheta_0 = (1/m) * sum(h - y)
      dJ/dtheta_1 = (1/m) * sum((h - y) * x)
      theta_j := theta_j - alpha * dJ/dtheta_j

    Limites: cuando x -> +inf, g(x) -> 1; cuando x -> -inf, g(x) -> 0
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
    eps = 1e-10

    for i in range(iterations):
        z = theta[0] + theta[1] * X_norm
        h = sigmoid(z)

        error = h - y

        grad_0 = (1 / m) * np.sum(error)
        grad_1 = (1 / m) * np.sum(error * X_norm)

        theta[0] -= alpha * grad_0
        theta[1] -= alpha * grad_1

        cost = -(1 / m) * np.sum(
            y * np.log(h + eps) + (1 - y) * np.log(1 - h + eps)
        )

        predictions = sigmoid(theta[0] + theta[1] * X_norm)

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
            predictions=predictions.tolist(),
            extra={
                'sigmoid_values': predictions.tolist(),
                'probabilities': predictions.tolist()
            }
        ))

    theta_real = [
        float(theta[0] - theta[1] * (x_mean / x_std)),
        float(theta[1] / x_std)
    ]

    return theta_real, history, float(x_mean), float(x_std)


def kmeans(
    X: List[float],
    y: List[float],
    k: int = 3,
    iterations: int = 100
) -> Tuple[List[float], List[IterationStep], float, float]:
    """
    K-Means clustering.

    Formulas:
      J = sum ||x_i - mu_c(i)||^2
      Asignar c(i) = argmin_j ||x_i - mu_j||^2
      Actualizar mu_j = (1/|C_j|) * sum x_i

    Limites: convergencia cuando las iteraciones -> infinito,
    las asignaciones y centroides dejan de cambiar.
    """
    points = np.column_stack([np.array(X, dtype=float), np.array(y, dtype=float)])
    m = len(points)

    np.random.seed(42)
    indices = np.random.choice(m, k, replace=False)
    centroids = points[indices].copy()

    history = []
    prev_assignments = None

    for i in range(iterations):
        distances = np.array([
            np.sqrt(np.sum((points - centroids[j])**2, axis=1))
            for j in range(k)
        ])
        assignments = np.argmin(distances, axis=0)

        cost = np.sum(np.min(distances, axis=0)) / m

        new_centroids = centroids.copy()
        for j in range(k):
            mask = assignments == j
            if np.sum(mask) > 0:
                new_centroids[j] = np.mean(points[mask], axis=0)

        shifted = np.array([
            np.sum((points - new_centroids[assignments[p]])**2)
            for p in range(m)
        ])
        inertia = float(np.sum(shifted))

        predictions = np.zeros(m)
        for p in range(m):
            predictions[p] = float(assignments[p])

        colors = []
        for p in range(m):
            colors.append(float(assignments[p]))

        history.append(IterationStep(
            iteration=i + 1,
            theta_0=float(centroids[0, 0]) if k > 0 else 0,
            theta_1=float(centroids[0, 1]) if k > 0 else 0,
            theta_0_real=float(new_centroids[0, 0]) if k > 0 else 0,
            theta_1_real=float(new_centroids[0, 1]) if k > 0 else 0,
            cost=float(cost),
            gradient_0=float(inertia),
            gradient_1=float(0),
            predictions=colors,
            extra={
                'centroids': new_centroids.tolist(),
                'assignments': assignments.tolist(),
                'k': k,
                'inertia': inertia,
                'converged': bool(prev_assignments is not None and np.array_equal(assignments, prev_assignments))
            }
        ))

        if prev_assignments is not None and np.array_equal(assignments, prev_assignments):
            break

        prev_assignments = assignments.copy()
        centroids = new_centroids

    return [float(centroids[0, 0]), float(centroids[0, 1])], history, 0.0, 1.0


def neural_network(
    X: List[float],
    y: List[float],
    alpha: float = 0.1,
    iterations: int = 100,
    hidden_size: int = 4
) -> Tuple[List[float], List[IterationStep], float, float]:
    """
    Red neuronal simple: 1 capa oculta con backpropagation.

    Arquitectura: input(1) -> hidden(hidden_size) -> output(1)
    Activacion: sigmoide

    Forward:
      z1 = W1 * x + b1
      a1 = g(z1)
      z2 = W2 * a1 + b2
      a2 = g(z2) = prediccion

    Backpropagation (regla de la cadena):
      delta2 = a2 - y
      delta1 = (W2^T * delta2) * g'(z1)
      dJ/dW2 = delta2 * a1^T
      dJ/dW1 = delta1 * x^T

    Derivadas: regla de la cadena para propagar el error hacia atras.
    Limites: sigmoide tiende a 0 o 1 cuando z -> +/-inf.
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    m = len(y)

    x_mean, x_std = X.mean(), X.std()
    if x_std == 0:
        x_std = 1
    X_norm = (X - x_mean) / x_std
    y_mean, y_std = y.mean(), y.std()
    if y_std == 0:
        y_std = 1
    y_norm = (y - y_mean) / y_std

    np.random.seed(42)
    W1 = np.random.randn(hidden_size, 1) * 0.5
    b1 = np.zeros((hidden_size, 1))
    W2 = np.random.randn(1, hidden_size) * 0.5
    b2 = np.zeros((1, 1))

    history = []
    eps = 1e-10

    for i in range(iterations):
        X_col = X_norm.reshape(1, -1)
        y_col = y_norm.reshape(1, -1)

        z1 = W1 @ X_col + b1
        a1 = sigmoid(z1)
        z2 = W2 @ a1 + b2
        a2 = sigmoid(z2)

        error = a2 - y_col
        cost = float((1 / (2 * m)) * np.sum(error ** 2))

        delta2 = error * a2 * (1 - a2)
        dW2 = (1 / m) * (delta2 @ a1.T)
        db2 = (1 / m) * np.sum(delta2, axis=1, keepdims=True)

        delta1 = (W2.T @ delta2) * (a1 * (1 - a1))
        dW1 = (1 / m) * (delta1 @ X_col.T)
        db1 = (1 / m) * np.sum(delta1, axis=1, keepdims=True)

        W1 -= alpha * dW1
        b1 -= alpha * db1
        W2 -= alpha * dW2
        b2 -= alpha * db2

        predictions_norm = sigmoid(W2 @ sigmoid(W1 @ X_col + b1) + b2)
        predictions_norm_flat = predictions_norm.flatten()

        predictions_real = predictions_norm_flat * y_std + y_mean

        grad_magnitude = float(np.sqrt(np.sum(dW1**2) + np.sum(dW2**2)))

        history.append(IterationStep(
            iteration=i + 1,
            theta_0=float(b2[0, 0]),
            theta_1=float(W2[0, 0]),
            theta_0_real=float(predictions_real.mean()),
            theta_1_real=float(predictions_real.std()),
            cost=cost,
            gradient_0=float(grad_magnitude),
            gradient_1=float(np.sum(np.abs(dW2))),
            predictions=predictions_real.tolist(),
            extra={
                'hidden_size': hidden_size,
                'weights_norm': float(np.sqrt(np.sum(W1**2) + np.sum(W2**2))),
                'predictions': predictions_real.tolist()
            }
        ))

    final_predictions = (sigmoid(W2 @ sigmoid(W1 @ X_norm.reshape(1, -1) + b1) + b2) * y_std + y_mean).flatten()

    theta = (
        W1.flatten().tolist()
        + b1.flatten().tolist()
        + W2.flatten().tolist()
        + b2.flatten().tolist()
    )

    return theta, history, float(x_mean), float(x_std)