import numpy as np
from typing import Callable
from src.utils import d_cross_entropy, d_sigmoid, sgd, sigmoid, cross_entropy


class MLP:
    def __init__(self) -> None:
        self.act_funcs = {"sigmoid": [sigmoid, d_sigmoid]}
        self.loss_funcs = {"cross_entropy": [cross_entropy, d_cross_entropy]}
        self.optimizers = {"sgd": sgd}
        self.params = {"batch_size": 1, "optimizer": "sgd", "lr": 0.001, "epochs": 100}
        self.loss_func = None
        self.layers = []

    def add_layer(
        self,
        n_in: int,
        n_neur: int,
        act_func: str,
    ) -> None:
        if act_func not in self.act_funcs.keys():
            raise ValueError("Invalid activation function name")
        self.layers.append(Layer(n_in, n_neur, self.act_funcs[act_func][0]))
        self.L_afd = self.act_funcs[act_func][1]

    def configure_training(self, loss_func: str, **kwargs) -> None:
        if loss_func not in self.loss_funcs.keys():
            raise ValueError("Invalid loss function name")

        self.loss_func = self.loss_funcs[loss_func][0]

        for key in kwargs:
            if key not in self.params.keys():
                raise ValueError(f"Invalid param key: {key}")

        self.params.update(kwargs)

    def propagate_back(
        self, batch_Y: np.ndarray, batch_X: np.ndarray, batch_Z: np.ndarray
    ) -> None:
        if not self.loss_func:
            raise RuntimeError(
                "Loss function not set. Use configure_training method first"
            )

        lf = self.loss_func
        afd = self.L_afd
        err_L = lf(batch_Y, batch_X) * afd(batch_Z)
        opt = self.params["optimizer"]
        lr = self.params["lr"]
        prev_err = err_L

        for layer in self.layers[::-1]:
            # next layer error
            next_err = (layer.weights.T @ prev_err) * afd()

    def train(self, x_train, y_train, x_test, y_test) -> None: ...

    def predict(self) -> None: ...


class Layer:
    def __init__(
        self,
        n_in: int,
        n_neur: int,
        act_func: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        self.weights = np.random.normal(0, 1, size=(n_neur, n_in))
        self.bias = np.zeros(n_neur)
        self.act_func = act_func

    def forward(self, x: np.ndarray) -> np.ndarray:
        z = np.matmul(self.weights, x) + self.bias
        return self.act_func(z)
