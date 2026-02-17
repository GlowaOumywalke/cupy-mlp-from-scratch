import cupy as cp


def sigmoid(x: float | cp.ndarray) -> cp.ndarray:
    return 1 / (1 + cp.exp(-x))


def d_sigmoid(x: float | cp.ndarray) -> float | cp.ndarray:
    s = sigmoid(x)
    return s * (1 - s)


def softmax(x: cp.ndarray) -> cp.ndarray:
    dem = cp.sum([cp.exp(xi) for xi in x])
    return cp.exp(x) / dem


def mat_softmax(X: cp.ndarray) -> cp.ndarray:
    dem = cp.sum(cp.exp(X), axis=1, keepdims=True)
    return cp.exp(X) / dem


def cross_entropy(Y: cp.ndarray, X: cp.ndarray) -> float:
    """
    Y - one-hot expected, Y.shape = (batch_size, n_classes)
    X - predictions,      X.shape = (batch_size, n_classes)
    """
    X = mat_softmax(X)
    eps = 1e-10
    X = cp.clip(X, eps, 1)
    return cp.mean(-cp.sum(Y * cp.log(X), axis=1))


def d_cross_entropy(Y: cp.ndarray, X: cp.ndarray) -> cp.ndarray:
    X = mat_softmax(X)
    return (X - Y) / Y.shape[0]


def sgd(X: cp.ndarray, X_grad: cp.ndarray, lr: float) -> cp.ndarray:
    return X - lr * X_grad
