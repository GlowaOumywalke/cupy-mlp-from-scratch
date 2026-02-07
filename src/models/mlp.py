import numpy as np
from src.utils import linear_comb, sigmoid, cross_entropy
from typing import Callable


class MLP:
    def __init__(self) -> None:
        self.act_func = sigmoid
        self.loss_func = cross_entropy
        self.layers = []

    def set_act_fun(self, f: Callable[[float], float]) -> None:
        self.act_func = f

    def set_loss_func(self, f: Callable[[float], float]) -> None:
        self.loss_func = f

    def add_layer(self, n_in: int, n_neur: int) -> None:
        self.layers.append(Layer(n_in, n_neur))

    def propagate_back(self, mu: float = 0.05) -> None:
        for layer in self.layers:
            ...

    def train(self, x_train, y_train, x_test, y_test) -> None: ...

    def predict(self) -> None: ...


class Layer:
    def __init__(self, n_in: int, n_neur) -> None:
        self.weights = np.random.normal(n_in, n_neur)
        self.bias = np.random.normal(n_neur)
        self.act_func = sigmoid

    def set_act_fun(self, f: Callable[[float], float]) -> None:
        self.act_func = f

    def forward(self, x: np.ndarray) -> float:
        z = linear_comb(x, self.weights) + self.weights
        return self.act_func(z)
