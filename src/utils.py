import numpy as np


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x: float) -> float:
    s = sigmoid(x)
    return s * (s - 1)


def cross_entropy(y: np.ndarray, p: np.ndarray) -> float:
    return -np.sum(y * np.log(p))


def linear_comb(w: np.ndarray, x: np.ndarray) -> float:
    return np.dot(w, x)
