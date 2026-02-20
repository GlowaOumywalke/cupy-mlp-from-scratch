import cupy as cp


def one_hot(labels: cp.ndarray, c: int) -> cp.ndarray:
    m = len(labels)
    res = cp.zeros((m, c))
    res[cp.arange(m), labels] = 1
    return res


def normalize_mnist(
    x_train: cp.ndarray, x_test: cp.ndarray
) -> tuple[cp.ndarray, cp.ndarray]:
    return x_train / 255, x_test / 255


def mnist_to_long(
    x_train: cp.ndarray, x_test: cp.ndarray
) -> tuple[cp.ndarray, cp.ndarray]:
    return x_train.reshape(x_train.shape[0], -1), x_test.reshape(x_test.shape[0], -1)
