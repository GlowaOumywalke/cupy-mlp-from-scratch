import numpy as np
import cupy as cp


# wow
def one_hot(labels: cp.ndarray, c: int) -> cp.ndarray:
    m = labels.shape[0]
    res = cp.zeros((m, c), dtype=cp.float32)
    res[cp.arange(m), labels] = 1
    return res


def normalize_mnist(
    x_train: np.ndarray, x_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    return x_train / 255, x_test / 255


def mnist_to_long(
    x_train: np.ndarray, x_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    return x_train.reshape(x_train.shape[0], -1), x_test.reshape(x_test.shape[0], -1)
