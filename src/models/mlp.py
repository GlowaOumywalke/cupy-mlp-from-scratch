import numpy as np
import cupy as cp
from tqdm import tqdm
from typing import Callable
from src.preprocessing import one_hot
from src.utils import (
    adam,
    d_cross_entropy,
    d_sigmoid,
    mat_softmax,
    relu,
    relu_d,
    sgd,
    sigmoid,
    cross_entropy,
)


class MLP:
    """
    Multi-Layer Perceptron (MLP) implementation using cupy for GPU acceleration.
    """

    def __init__(self) -> None:
        """
        Initializes the MLP with default parameters and empty layer list.

        Sets up supported activation functions, loss functions, and optimizers.
        """
        self.act_funcs = {
            "sigmoid": [sigmoid, d_sigmoid],
            "relu": [relu, relu_d],
            "linear": [lambda x: x, lambda x: np.ones_like(x)],
            "softmax": [mat_softmax, None],
        }
        self.loss_funcs = {"cross_entropy": [cross_entropy, d_cross_entropy]}
        self.optimizers = {"sgd": sgd, "adam": adam}
        self.params = {
            "batch_size": 1,
            "optimizer": "sgd",
            "lr": 0.001,
            "epochs": 100,
            "loss_func": "",
            "activation_last": "",
        }
        self.loss_func = None
        self.loss_func_d = None
        self.input_s = None
        self.layers = []

    def set_input_size(self, size: int) -> None:
        """
        Sets the input size of neural network.

        Args:
            size (int): Number of input features

        Raises:
            ValueError: If the input size has already been set

        Returns:
            None
        """
        if self.input_s is not None:
            raise ValueError("Input size is already set")
        self.input_s = size

    def add_layer(
        self,
        n_neurons: int,
        act_func: str,
    ) -> None:
        """
        Adds new layer to the neural network.

        Args:
            n_neurons (int): Number of neurons in layer.
            act_func (str): Name of activation function.
                Supported: "sigmoid", "relu", "linear", "softmax".

        Raises:
            ValueError: If the activation function is not supported.
            RuntimeError: If input size is not set. Use `set_input_size()` first.

        Returns:
            None
        """
        if act_func not in self.act_funcs.keys():
            raise ValueError("Invalid activation function name")
        if self.input_s is None:
            raise RuntimeError("Input size not set. Use set_input_size method first")

        n_in = self.layers[-1].weights.shape[0] if self.layers else self.input_s

        self.layers.append(
            Layer(
                n_in,
                n_neurons,
                self.act_funcs[act_func][0],
                self.act_funcs[act_func][1],
                act_func,
            )
        )
        self.params["activation_last"] = act_func

    def configure_training(self, loss_function: str, **kwargs) -> None:
        """
        Configures the training parameters and loss function

        Args:
            loss_function (str): Name of the loss function.
                Supported: "cross_entropy".

            **kwargs: Optional training parameters:
                - batch_size (int): Size of the batch. (Default: 1)
                - optimizer (str): Name of the optimizer. (Default: "sgd")
                     Supported: "sgd", "adam"
                - lr (float): Learning rate. (Default: 0.001)
                - epochs (int): Number of training epochs. (Default: 100)

        Raises:
            ValueError: If an unsupported loss function name is provided.
            ValueError: If an unsupported keyword argument is provided.
            ValueError: If cross entropy is set as loss function and softmax is not last layer activation function.

        Returns:
            None
        """
        if loss_function not in self.loss_funcs.keys():
            raise ValueError("Invalid loss function name")
        for key in kwargs:
            if key not in self.params.keys():
                raise ValueError(f"Invalid param key: {key}")

        self.loss_func = self.loss_funcs[loss_function][0]
        self.loss_func_d = self.loss_funcs[loss_function][1]
        self.params["loss_func"] = loss_function

        self.params.update(kwargs)

        if (
            loss_function == "cross_entropy"
            and self.params["activation_last"] != "softmax"
        ):
            raise ValueError("Cross entropy requires softmax as last activation")

        if self.params["optimizer"] == "adam":
            self.adam_state = []
            for layer in self.layers:
                state = {
                    "m_w": cp.zeros_like(layer.weights),
                    "v_w": cp.zeros_like(layer.weights),
                    "m_b": cp.zeros_like(layer.bias),
                    "v_b": cp.zeros_like(layer.bias),
                    "t": 0,
                }
                self.adam_state.append(state)

    def _propagate_back(
        self,
        batch_Y: cp.ndarray,
        pred: cp.ndarray,
        zs: cp.ndarray,
        activations: cp.ndarray,
    ) -> None:
        """
        Perform backpropagation and update network parameters.

        Computes gradient of the loss function with respect to weights and biases
        using backpropagation alghortihm, then updated the parameters using the
        configured optimizer.

        Args:
            batch_Y (cp.ndarray): Ground-truth labels for the current batch.
                Shape: (batch_size, output_dim)
            pred (cp.ndarray): Model predictions from the forward pass.
                Shape: (batch_size, output_dim).
            zs (cp.ndarray): Linear combinations for each layer.
                Length equals number of layers.
            activations (cp.ndarray): Activations for each layer.
                Length equals number of layers.

        Raises:
            RuntimeError: If loss function is not set. `Use configure_training()` method first.

        Returns:
            None
        """
        if self.params["loss_func"] == "":
            raise RuntimeError(
                "Loss function not set. Use configure_training() method first"
            )

        loss_func_d = self.loss_funcs[self.params["loss_func"]][1]
        last_layer_act_d = self.layers[-1].act_func_d
        lr = self.params["lr"]

        if (
            self.params["loss_func"] == "cross_entropy"
            and self.params["activation_last"] == "softmax"
        ):
            delta = loss_func_d(batch_Y, pred)
        else:
            delta = loss_func_d(batch_Y, pred) * last_layer_act_d(zs[-1])

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            a_prev = activations[i]

            grad_b = cp.sum(delta, axis=0)
            grad_w = a_prev.T @ delta

            weights = layer.weights
            if self.params["optimizer"] == "adam":
                state = self.adam_state[i]
                state["t"] += 1
                layer.weights, state["m_w"], state["v_w"] = adam(
                    layer.weights,
                    grad_w.T,
                    self.params["lr"],
                    state["m_w"],
                    state["v_w"],
                    state["t"],
                )
                layer.bias, state["m_b"], state["v_b"] = adam(
                    layer.bias,
                    grad_b,
                    self.params["lr"],
                    state["m_b"],
                    state["v_b"],
                    state["t"],
                )
            if self.params["optimizer"] == "sgd":
                layer.weights = sgd(layer.weights, grad_w.T, lr)
                layer.bias = sgd(layer.bias, grad_b, lr)

            if i > 0:
                prev_layer = self.layers[i - 1]
                delta = (delta @ weights) * prev_layer.act_func_d(zs[i - 1])

    def train(
        self,
        x_train: cp.ndarray,
        y_train: cp.ndarray,
        x_test: cp.ndarray,
        y_test: cp.ndarray,
        save_dir=None,
    ) -> None:
        """
        Trains the configured model on the provided dataset.

        Args:
            x_train (cp.ndarray): Training data features.
            y_train (cp.ndarray): Training data labels/target values.
            x_test (cp.ndarray): Test data features for evaluation.
            y_test (cp.ndarray): Test data labels for evaluation.
            save_dir (str, optional): Directory path where model weights and biases
                will be saved. If None, saving is skipped.

        Raises:
            RuntimeError: If the loss function has not been set. Use `configure_training()` first.

        Returns:
            None
        """
        if self.params["loss_func"] == "":
            raise RuntimeError(
                "Loss function not set. Use configure_training() method first"
            )

        loss_func = self.loss_funcs[self.params["loss_func"]][0]

        bs = self.params["batch_size"]
        rng = np.random.default_rng()

        pbar = tqdm(range(self.params["epochs"]))
        for i in pbar:
            idx = rng.permutation(len(x_train))
            x_shuffled = x_train[idx]
            y_shuffled = y_train[idx]

            epoch_loss = 0
            num_batches = 0
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

                epoch_loss += loss_func(batch_Y, a)
                num_batches += 1
                self._propagate_back(batch_Y, a, zs, activations)

            curr_loss = epoch_loss / num_batches
            pbar.set_description(f"Epoch: {i + 1}, Avg Loss: {curr_loss:.4f}")

        self.test(x_test, y_test)

        if save_dir:
            layers_data = {
                f"layer_{i + 1}": {
                    "weights": self.layers[i].weights,
                    "bias": self.layers[i].bias,
                    "act": self.layers[i].act_func_name,
                }
                for i in range(len(self.layers))
            }
            cp.savez(f"{save_dir}model.npz", **layers_data)

    def test(self, x_test: cp.ndarray, y_test: cp.ndarray) -> None:
        """
        Tests the trained model and prints its accuracy score.

        Args:
            x_test (cp.ndarray): Features for the test dataset.
            y_test (cp.ndarray): Ground-truth labels for the test dataset.

        Returns:
            None
        """
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
        acc = cp.mean(cp.argmax(preds, axis=1) == cp.argmax(y_test, axis=1))
        print(f"Accuracy: {acc:.4f}")

    def predict(self, x: cp.ndarray, weights=None, bias=None) -> cp.ndarray:
        """
        Generates model predictions for the given input data.

        Optionally allows overriding current layer weights and biases before prediction.

        Args:
            x (cp.ndarray): Input data features.
            weights (list[cp.ndarray], optional): List of weight matrices for each layer.
            bias (list[cp.ndarray], optional): List of bias vectors for each layer.

        Raises:
            AssertionError: If provided weights length doesn't match the number of layers.

        Returns:
            cp.ndarray: The output of the model.
        """
        if weights and bias:
            assert len(weights) == len(self.layers), "Invalid weights size"
            for i in range(len(weights)):
                self.layers[i].weights = weights[i]
                self.layers[i].bias = bias[i]

        for layer in self.layers:
            x, _ = layer.forward(x)

        return x

    @classmethod
    def load_model(cls, file_path: str, loss_function: str) -> "MLP":
        """
        Loads a saved MLP model from a file.

        This method instantiates a new MLP, restores its layers, weights,
        biases, and activation functions based on the saved data.

        Args:
            file_path (str): Path to the saved model file.
            loss_function (str): Name of the loss function to be used for
                further training or evaluation.

        Returns:
            MLP: A fully initialized and populated instance of the MLP class.

        Note:
            The saved file must contain keys formatted as 'layer_1', 'layer_2', etc.,
            each storing a dictionary with 'weights', 'bias', and 'act'.
        """
        model = cls()

        saved_model = np.load(file_path, allow_pickle=True)

        first_l = saved_model["layer_1"].item()
        model.set_input_size(first_l["weights"].shape[1])
        for i in range(len(saved_model.files)):
            layer = saved_model[f"layer_{i + 1}"].item()
            model.add_layer(layer["weights"].shape[0], layer["act"])

        for i in range(len(model.layers)):
            layer = saved_model[f"layer_{i + 1}"].item()
            model.layers[i].weights = cp.array(layer["weights"])
            model.layers[i].bias = cp.array(layer["bias"])

        model.params["loss_func"] = loss_function
        model.loss_func = model.loss_funcs[loss_function][0]
        model.loss_func_d = model.loss_funcs[loss_function][1]
        model.params["activation_last"] = model.layers[-1].act_func_name

        return model


class Layer:
    def __init__(
        self,
        n_in: int,
        n_neur: int,
        act_func: Callable[[cp.ndarray], cp.ndarray],
        act_func_d: Callable[[cp.ndarray], cp.ndarray],
        act_func_name: str,
    ) -> None:
        """
        Initializes a neural network layer.

        Args:
            n_in (int): Number of input connections (features).
            n_neur (int): Number of neurons in this layer.
            act_func (Callable): Activation function.
            act_func_d (Callable): Derivative of the activation function.
            act_func_name (str): Name of the activation function. Used to save the model.
        """
        self.weights = cp.random.randn(n_neur, n_in) * cp.sqrt(2 / n_in)
        self.bias = cp.zeros(n_neur)
        self.act_func = act_func
        self.act_func_d = act_func_d
        self.act_func_name = act_func_name

    def forward(self, x: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
        """
        Performs a forward pass through the layer.

        Calculates the linear transformation (z = xW^T + b) followed by
        the activation function.

        Args:
            x (cp.ndarray): Input activations from the previous layer
                or input data of shape (batch_size, input_features).

        Returns:
            tuple[cp.ndarray, cp.ndarray]: A tuple containing:
                - a (cp.ndarray): Activated output, f(z).
                - z (cp.ndarray): Pre-activation linear result (stored for backprop).
        """
        z = x @ self.weights.T + self.bias
        return self.act_func(z), z
