import numpy as np
import cupy as cp
from tqdm import tqdm
from typing import Callable
from src.utils import d_cross_entropy, d_sigmoid, sgd, sigmoid, cross_entropy
from src.preprocessing import one_hot


class MLP:
    def __init__(self) -> None:
        self.act_funcs = {
            "sigmoid": [sigmoid, d_sigmoid],
            "linear": [lambda x: x, lambda x: np.ones_like(x)],
        }
        self.loss_funcs = {"cross_entropy": [cross_entropy, d_cross_entropy]}
        self.optimizers = {"sgd": sgd}
        self.params = {"batch_size": 1, "optimizer": "sgd", "lr": 0.001, "epochs": 100}
        self.loss_func = None
        self.loss_fund_d = None
        self.layers = []

    def add_layer(
        self,
        n_in: int,
        n_neur: int,
        act_func: str,
    ) -> None:
        if act_func not in self.act_funcs.keys():
            raise ValueError("Invalid activation function name")
        self.layers.append(
            Layer(
                n_in, n_neur, self.act_funcs[act_func][0], self.act_funcs[act_func][1]
            )
        )
        self.L_afd = self.act_funcs[act_func][1]

    def configure_training(self, loss_func: str, **kwargs) -> None:
        if loss_func not in self.loss_funcs.keys():
            raise ValueError("Invalid loss function name")

        self.loss_func = self.loss_funcs[loss_func][0]
        self.loss_func_d = self.loss_funcs[loss_func][1]

        for key in kwargs:
            if key not in self.params.keys():
                raise ValueError(f"Invalid param key: {key}")

        self.params.update(kwargs)

    def propagate_back(
        self,
        batch_Y: cp.ndarray,
        pred: cp.ndarray,
        zs: cp.ndarray | list,
        activations: cp.ndarray | list,
    ) -> None:
        if not self.loss_func_d:
            raise RuntimeError(
                "Loss function not set. Use configure_training method first"
            )

        lfd = self.loss_func_d
        opt = self.optimizers[self.params["optimizer"]]
        lr = self.params["lr"]

        delta = lfd(batch_Y, pred)
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            a_prev = activations[i]

            grad_b = cp.sum(delta, axis=0)
            grad_w = a_prev.T @ delta

            weights = layer.weights.copy()
            layer.weights = opt(layer.weights, grad_w.T, lr)
            layer.bias = opt(layer.bias, grad_b, lr)

            if i > 0:
                prev_layer = self.layers[i - 1]
                delta = (delta @ weights) * prev_layer.act_func_d(zs[i - 1])

    def train(self, x_train, y_train, x_test, y_test, save_dir=None) -> None:
        if self.loss_func is None:
            raise RuntimeError(
                "Loss function not set. Use configure_training method first"
            )

        bs = self.params["batch_size"]
        rng = np.random.default_rng()

        for i in tqdm(range(self.params["epochs"])):
            idx = rng.permutation(len(x_train))
            x_shuffled = x_train[idx]
            y_shuffled = y_train[idx]
            curr_loss = None

            for j in range(0, len(x_train), bs):
                batch_X = x_shuffled[j : j + bs]
                batch_Y = y_shuffled[j : j + bs]

                a = batch_X
                activations = [a]
                zs = []

                for layer in self.layers:
                    a, z = layer.forward(a)
                    zs.append(z)
                    activations.append(a)

                curr_loss = self.loss_func(batch_Y, a)
                self.propagate_back(batch_Y, a, zs, activations)

            if i % 10 == 0:
                print("--" * 20)
                print(f"Current loss: {curr_loss:.4f}")
                print("--" * 20)

        self.test(x_test, y_test)
        self.layers_weight = {
            f"layer{i}": self.layers[i].weights for i in range(len(self.layers))
        }
        self.layers_bias = {
            f"layer{i}": self.layers[i].bias for i in range(len(self.layers))
        }
        if save_dir:
            cp.savez(f"{save_dir}weights.npz", **self.layers_weight)
            cp.savez(f"{save_dir}bias.npz", **self.layers_bias)

    def test(self, x_test: cp.ndarray, y_test: cp.ndarray) -> None:
        preds = []
        bs = self.params["batch_size"]
        for i in range(0, len(x_test), bs):
            batch_X = x_test[i : i + bs]
            a = batch_X
            for layer in self.layers:
                a, _ = layer.forward(a)

            pred = cp.argmax(a, axis=1)
            preds.append(pred)

        preds = cp.concatenate(preds)
        preds = one_hot(preds, 10)
        acc = cp.mean(preds == y_test)
        print(f"Accuracy: {acc:.4f}")

    def predict(self, x: cp.ndarray, weights=None, bias=None) -> cp.ndarray:
        if weights and bias:
            assert len(weights) == len(self.layers), "Invalid weights size"
            for i in range(len(weights)):
                self.layers[i].weights = weights[i]
                self.layers[i].bias = bias[i]

        for layer in self.layers:
            x, _ = layer.forward(x)

        return x


class Layer:
    def __init__(
        self,
        n_in: int,
        n_neur: int,
        act_func: Callable[[cp.ndarray], cp.ndarray],
        act_func_d: Callable[[cp.ndarray], cp.ndarray],
    ) -> None:
        self.weights = cp.random.normal(0, 1, size=(n_neur, n_in))
        self.bias = cp.zeros(n_neur)
        self.act_func = act_func
        self.act_func_d = act_func_d

    def forward(self, x: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
        z = x @ self.weights.T + self.bias
        return self.act_func(z), z
