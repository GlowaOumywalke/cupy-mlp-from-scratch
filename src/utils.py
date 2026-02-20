import cupy as cp


def sigmoid(x: float | cp.ndarray) -> cp.ndarray:
    """Computes the sigmoid activation function: $f(x) = 1 / (1 + e^{-x})$."""
    return 1 / (1 + cp.exp(-x))


def d_sigmoid(x: float | cp.ndarray) -> float | cp.ndarray:
    """Computes the derivative of the sigmoid function for backpropagation."""
    s = sigmoid(x)
    return s * (1 - s)


def relu(x):
    """Computes the Rectified Linear Unit: $f(x) = max(0, x)$."""
    return cp.maximum(0, x)


def relu_d(x):
    """Computes the gradient of ReLU (1 for $x > 0$, else 0)."""
    return (x > 0).astype(cp.float32)


def mat_softmax(X: cp.ndarray) -> cp.ndarray:
    """
    Computes Softmax for a batch of input vectors.

    Uses the 'max trick' for numerical stability to prevent overflow.

    Args:
        X (cp.ndarray): Input matrix of shape (batch_size, n_classes).
    Returns:
        cp.ndarray: Probability distribution for each row.
    """
    X = X - cp.max(X, axis=1, keepdims=True)
    X_exp = cp.exp(X)
    dem = cp.sum(X_exp, axis=1, keepdims=True)
    return X_exp / dem


def cross_entropy(Y: cp.ndarray, X: cp.ndarray) -> float:
    """
    Computes Categorical Cross-Entropy loss.

    Args:
        Y (cp.ndarray): One-hot encoded true labels. Shape: (batch, classes).
        X (cp.ndarray): Predicted probabilities (after Softmax). Shape: (batch, classes).
    Returns:
        float: Mean loss value for the batch.
    """
    eps = 1e-10
    X = cp.clip(X, eps, 1)
    return cp.mean(-cp.sum(Y * cp.log(X), axis=1))


def d_cross_entropy(Y: cp.ndarray, X: cp.ndarray) -> cp.ndarray:
    """
    Gradient of Cross-Entropy loss with respect to Softmax input.

    Note: This simplified version (X - Y) assumes Softmax was the last activation.
    """
    return (X - Y) / Y.shape[0]


def sgd(X: cp.ndarray, X_grad: cp.ndarray, lr: float) -> cp.ndarray:
    """Performs a simple Stochastic Gradient Descent update."""
    return X - lr * X_grad


def adam(
    X: cp.ndarray, X_grad: cp.ndarray, lr: float, m: cp.ndarray, v: cp.ndarray, t: int
) -> cp.ndarray:
    """
    Performs Adam (Adaptive Moment Estimation) update.

    Args:
        X (cp.ndarray): Parameters (weights or biases) to update.
        X_grad (cp.ndarray): Gradient of the parameters.
        lr (float): Learning rate.
        m (cp.ndarray): First moment vector (moving average of gradients).
        v (cp.ndarray): Second moment vector (moving average of squared gradients).
        t (int): Current time step (iteration).

    Returns:
        tuple: (Updated parameters, updated m, updated v).
    """
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    m = beta1 * m + (1 - beta1) * X_grad
    v = beta2 * v + (1 - beta2) * X_grad**2

    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)

    X_new = X - lr * m_hat / (cp.sqrt(v_hat) + eps)

    return X_new, m, v
