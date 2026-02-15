import numpy as np


def sigmoid(x: float | np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    s = sigmoid(x)
    return s * (s - 1)


def softmax(x: np.ndarray) -> np.ndarray:
    dem = np.sum([np.exp(xi) for xi in x])
    return np.exp(x) / dem


def mat_softmax(X: np.ndarray) -> np.ndarray:
    dem = np.sum(np.exp(X), axis=1, keepdims=True)
    return np.exp(X) / dem


def cross_entropy(Y: np.ndarray, X: np.ndarray) -> float:
    """
    Y - one-hot expected, Y.shape = (batch_size, n_classes)
    X - predictions,      X.shape = (batch_size, n_classes)
    """
    X = mat_softmax(X)
    eps = 1e-10
    X = np.clip(X, eps, 1)
    return np.mean(-np.sum(Y * np.log(X), axis=1))


def d_cross_entropy(Y: np.ndarray, X: np.ndarray) -> np.ndarray:
    X = mat_softmax(X)
    return (X - Y) / Y.shape[0]


def sgd(X: np.ndarray, X_grad: np.ndarray, lr: float) -> np.ndarray:
    return X - lr * X_grad
