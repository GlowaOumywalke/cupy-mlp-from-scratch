import numpy as np
import cupy as cp
import argparse as ap
from src.models.mlp import MLP
from src.preprocessing import mnist_to_long, normalize_mnist, one_hot


def main():
    p = ap.ArgumentParser()
    p.add_argument(
        "--p",
        "--path",
        type=str,
        help="path to the npz file with the training and test dataset",
    )
    args = p.parse_args()
    path = args.p

    with np.load(path) as data:
        x_train = cp.asarray(data["x_train"])
        y_train = cp.asarray(data["y_train"])
        x_test = cp.asarray(data["x_test"])
        y_test = cp.asarray(data["y_test"])

    y_train_oh = one_hot(y_train, 10)
    y_test_oh = one_hot(y_test, 10)

    x_train, x_test = normalize_mnist(x_train, x_test)
    x_train, x_test = mnist_to_long(x_train, x_test)

    mlp = MLP()
    mlp.add_layer(28 * 28, 128, "sigmoid")
    mlp.add_layer(128, 64, "sigmoid")
    mlp.add_layer(64, 10, "linear")
    mlp.configure_training("cross_entropy", epochs=1000, batch_size=128, lr=0.01)
    mlp.train(x_train, y_train_oh, x_test, y_test_oh, "model/")


if __name__ == "__main__":
    main()
