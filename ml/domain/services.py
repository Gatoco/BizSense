from typing import List, Tuple
import numpy as np
from ml.domain.models import IterationStep

EPS = 1e-10


def gradient_descent(
    X: List[float],
    y: List[float],
    alpha: float = 0.01,
    iterations: int = 100,
    lambda_reg: float = 0.0,
    early_stopping: bool = True,
    tol: float = 1e-6
) -> Tuple[List[float], List[IterationStep], float, float]:
    """
    Gradiente descendente para regresion lineal simple con early stopping y L2.

    Formulas:
      h(x) = theta_0 + theta_1 * x
      J(theta) = (1/2m) * sum(h(x_i) - y_i)^2 + (lambda/2m) * sum(theta_j^2)
      dJ/dtheta_0 = (1/m) * sum(h(x_i) - y_i)
      dJ/dtheta_1 = (1/m) * sum(h(x_i) - y_i) * x_i + (lambda/m) * theta_1
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
        grad_1 = (1 / m) * np.sum(error * X_norm) + (lambda_reg / m) * theta[1]

        theta[0] -= alpha * grad_0
        theta[1] -= alpha * grad_1

        cost = (1 / (2 * m)) * (np.sum(error ** 2) + (lambda_reg / 2) * np.sum(theta ** 2))

        predictions = theta[0] + theta[1] * X_norm

        theta_0_real = float(theta[0] - theta[1] * (x_mean / x_std))
        theta_1_real = float(theta[1] / x_std)

        grad_norm = np.sqrt(grad_0**2 + grad_1**2)

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

        # ponytail: early stopping if gradient magnitude is small
        if early_stopping and grad_norm < tol:
            break

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
    iterations: int = 100,
    lambda_reg: float = 0.0,
    early_stopping: bool = True,
    tol: float = 1e-6
) -> Tuple[List[float], List[IterationStep], float, float]:
    """
    Regresion logistica con gradiente descendente, L2 regularization y early stopping.

    Formulas:
      g(z) = 1 / (1 + e^-z)
      h(x) = g(theta_0 + theta_1 * x)
      J(theta) = -(1/m) * sum(y * log(h) + (1-y) * log(1-h)) + (lambda/2m) * sum(theta^2)
      dJ/dtheta_0 = (1/m) * sum(h - y)
      dJ/dtheta_1 = (1/m) * sum((h - y) * x) + (lambda/m) * theta_1

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

    for i in range(iterations):
        z = theta[0] + theta[1] * X_norm
        h = sigmoid(z)

        error = h - y

        grad_0 = (1 / m) * np.sum(error)
        grad_1 = (1 / m) * np.sum(error * X_norm) + (lambda_reg / m) * theta[1]

        theta[0] -= alpha * grad_0
        theta[1] -= alpha * grad_1

        cost = -(1 / m) * np.sum(
            y * np.log(h + EPS) + (1 - y) * np.log(1 - h + EPS)
        ) + (lambda_reg / (2 * m)) * np.sum(theta ** 2)

        predictions = sigmoid(theta[0] + theta[1] * X_norm)

        theta_0_real = float(theta[0] - theta[1] * (x_mean / x_std))
        theta_1_real = float(theta[1] / x_std)

        grad_norm = np.sqrt(grad_0**2 + grad_1**2)

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
            extra={'probabilities': predictions.tolist()}
        ))

        # ponytail: early stopping
        if early_stopping and grad_norm < tol:
            break

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
    K-Means clustering con inicializacion K-means++.

    Formulas:
      J = sum ||x_i - mu_c(i)||^2
      Asignar c(i) = argmin_j ||x_i - mu_j||^2
      Actualizar mu_j = (1/|C_j|) * sum x_i

    Limites: convergencia cuando las iteraciones -> infinito,
    las asignaciones y centroides dejan de cambiar.
    """
    points = np.column_stack([np.array(X, dtype=float), np.array(y, dtype=float)])
    m = len(points)

    # ponytail: K-means++ initialization
    centroids = np.zeros((k, 2))
    centroids[0] = points[np.random.randint(m)]
    for c in range(1, k):
        distances = np.array([
            np.min([np.sum((points[i] - centroids[j])**2) for j in range(c)])
            for i in range(m)
        ])
        probs = distances / distances.sum()
        centroids[c] = points[np.random.choice(m, p=probs)]

    history = []
    prev_assignments = None

    for i in range(iterations):
        distances = np.array([
            np.sqrt(np.sum((points - centroids[j])**2, axis=1))
            for j in range(k)
        ])
        assignments = np.argmin(distances, axis=0)

        cost = np.sum(np.min(distances, axis=0)) / m

        converged = prev_assignments is not None and np.array_equal(assignments, prev_assignments)

        history.append(IterationStep(
            iteration=i + 1,
            theta_0=float(centroids[0, 0]),
            theta_1=float(centroids[0, 1]),
            theta_0_real=float(centroids[0, 0]),
            theta_1_real=float(centroids[0, 1]),
            cost=float(cost),
            gradient_0=float(cost * m),
            gradient_1=float(0),
            predictions=assignments.astype(float).tolist(),
            extra={
                'centroids': centroids.tolist(),
                'assignments': assignments.tolist(),
                'k': k,
                'inertia': float(cost * m),
                'converged': converged
            }
        ))

        if converged:
            break

        prev_assignments = assignments.copy()

        new_centroids = centroids.copy()
        for j in range(k):
            mask = assignments == j
            if np.sum(mask) > 0:
                new_centroids[j] = np.mean(points[mask], axis=0)
        centroids = new_centroids

    theta = centroids.flatten().tolist()
    return theta, history, 0.0, 1.0


def neural_network(
    X: List[float],
    y: List[float],
    alpha: float = 0.1,
    iterations: int = 100,
    hidden_size: int = 4,
    lambda_reg: float = 0.0,
    momentum: float = 0.9,
    early_stopping: bool = True,
    tol: float = 1e-6
) -> Tuple[List[float], List[IterationStep], float, float]:
    """
    Red neuronal simple: 1 capa oculta con backpropagation, L2, momentum y early stopping.

    Arquitectura: input(1) -> hidden(hidden_size) -> output(1)
    Activacion: sigmoide (hidden) + identidad (output) para regresión

    Forward:
      z1 = W1 * x + b1
      a1 = g(z1)
      z2 = W2 * a1 + b2
      a2 = z2 = prediccion (identidad, no sigmoide)

    Backpropagation (regla de la cadena):
      delta2 = a2 - y
      delta1 = (W2^T * delta2) * g'(z1)
      dJ/dW2 = delta2 * a1^T + (lambda/m) * W2
      dJ/dW1 = delta1 * x^T + (lambda/m) * W1

    Derivadas: regla de la cadena para propagar el error hacia atras.
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

    W1 = np.random.randn(hidden_size, 1) * 0.5
    b1 = np.zeros((hidden_size, 1))
    W2 = np.random.randn(1, hidden_size) * 0.5
    b2 = np.zeros((1, 1))

    # ponytail: momentum velocities
    v_W1 = np.zeros_like(W1)
    v_b1 = np.zeros_like(b1)
    v_W2 = np.zeros_like(W2)
    v_b2 = np.zeros_like(b2)

    history = []

    for i in range(iterations):
        X_col = X_norm.reshape(1, -1)
        y_col = y_norm.reshape(1, -1)

        z1 = W1 @ X_col + b1
        a1 = sigmoid(z1)
        z2 = W2 @ a1 + b2
        a2 = z2  # ponytail: identidad en output para regresión, no sigmoide

        error = a2 - y_col
        cost = float((1 / (2 * m)) * np.sum(error ** 2))

        l2_term = (lambda_reg / (2 * m)) * (np.sum(W1**2) + np.sum(W2**2))

        delta2 = error  # ponytail: gradiente de salida para identidad
        dW2 = (1 / m) * (delta2 @ a1.T) + (lambda_reg / m) * W2
        db2 = (1 / m) * np.sum(delta2, axis=1, keepdims=True)

        delta1 = (W2.T @ delta2) * (a1 * (1 - a1))
        dW1 = (1 / m) * (delta1 @ X_col.T) + (lambda_reg / m) * W1
        db1 = (1 / m) * np.sum(delta1, axis=1, keepdims=True)

        # ponytail: momentum update
        v_W1 = momentum * v_W1 - alpha * dW1
        v_b1 = momentum * v_b1 - alpha * db1
        v_W2 = momentum * v_W2 - alpha * dW2
        v_b2 = momentum * v_b2 - alpha * db2

        W1 += v_W1
        b1 += v_b1
        W2 += v_W2
        b2 += v_b2

        predictions_norm = W2 @ sigmoid(W1 @ X_col + b1) + b2
        predictions_norm_flat = predictions_norm.flatten()

        predictions_real = predictions_norm_flat * y_std + y_mean

        grad_magnitude = float(np.sqrt(np.sum(dW1**2) + np.sum(dW2**2)))

        history.append(IterationStep(
            iteration=i + 1,
            theta_0=float(b2[0, 0]),
            theta_1=float(W2[0, 0]),
            theta_0_real=float(predictions_real.mean()),
            theta_1_real=float(predictions_real.std()),
            cost=cost + l2_term,
            gradient_0=float(grad_magnitude),
            gradient_1=float(np.sum(np.abs(dW2))),
            predictions=predictions_real.tolist(),
            extra={
                'hidden_size': hidden_size,
                'weights_norm': float(np.sqrt(np.sum(W1**2) + np.sum(W2**2))),
                'predictions': predictions_real.tolist(),
                'x_mean': float(x_mean),
                'x_std': float(x_std),
                'y_mean': float(y_mean),
                'y_std': float(y_std)
            }
        ))

        # ponytail: early stopping
        if early_stopping and grad_magnitude < tol:
            break

    final_predictions = (W2 @ sigmoid(W1 @ X_norm.reshape(1, -1) + b1) + b2 * y_std + y_mean).flatten()

    theta = (
        W1.flatten().tolist()
        + b1.flatten().tolist()
        + W2.flatten().tolist()
        + b2.flatten().tolist()
    )

    return theta, history, float(x_mean), float(x_std)
