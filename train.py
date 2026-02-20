"""
MNIST Training Script using Custom MLP Framework.

Usage:
    python main.py --path data/mnist.npz

This script loads the MNIST dataset, preprocesses it (normalization and reshaping),
constructs a 3-layer MLP architecture, and starts the training process using GPU acceleration.
"""

import os
import numpy as np
import cupy as cp
import argparse as ap
from src.models.mlp import MLP
from src.preprocessing import mnist_to_long, normalize_mnist, one_hot


def main():
    p = ap.ArgumentParser(description="Train a MLP model on a provided NPZ dataset.")
    p.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the .npz file containing 'x_train', 'y_train', 'x_test', 'y_test'",
    )
    args = p.parse_args()
    path = args.p

    # data loading
    with np.load(path) as data:
        x_train = cp.asarray(data["x_train"])
        y_train = cp.asarray(data["y_train"])
        x_test = cp.asarray(data["x_test"])
        y_test = cp.asarray(data["y_test"])

    # one-hot encoding for cross entropy loss
    y_train_oh = one_hot(y_train, 10)
    y_test_oh = one_hot(y_test, 10)

    # normalizing and flattening 28x28 images to 784 vectors
    x_train, x_test = normalize_mnist(x_train, x_test)
    x_train, x_test = mnist_to_long(x_train, x_test)

    # model architecture
    mlp = MLP()
    mlp.set_input_size(28 * 28)
    mlp.add_layer(128, "relu")
    mlp.add_layer(64, "relu")
    mlp.add_layer(10, "softmax")
    mlp.configure_training(
        "cross_entropy", epochs=100, optimizer="adam", batch_size=128, lr=0.001
    )

    os.makedirs("model", exist_ok=True)
    # training
    cp.random.seed(67)
    mlp.train(x_train, y_train_oh, x_test, y_test_oh, "model/")


if __name__ == "__main__":
    main()
