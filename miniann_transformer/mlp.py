from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .activations import GELU, ReLU, Sigmoid, Tanh
from .layers import Linear
from .tensor import Array, Module, Parameter, count_parameters


ActivationName = Literal["relu", "gelu", "tanh", "sigmoid"]


def make_activation(name: ActivationName) -> ReLU | GELU | Tanh | Sigmoid:
    if name == "relu":
        return ReLU()
    if name == "gelu":
        return GELU()
    if name == "tanh":
        return Tanh()
    if name == "sigmoid":
        return Sigmoid()
    raise ValueError(f"unsupported activation: {name}")


@dataclass
class MLPConfig:
    input_dim: int = 2
    hidden_dims: tuple[int, ...] = (8, 8)
    output_dim: int = 1
    activation: ActivationName = "tanh"
    output_activation: ActivationName | None = "sigmoid"


class MeanSquaredError:
    def __init__(self) -> None:
        self.prediction: Array | None = None
        self.target: Array | None = None

    def forward(self, prediction: Array, target: Array) -> float:
        self.prediction = prediction
        self.target = target
        diff = prediction - target
        return float(np.mean(diff * diff))

    def backward(self) -> Array:
        if self.prediction is None or self.target is None:
            raise RuntimeError("MeanSquaredError.backward called before forward")
        return 2.0 * (self.prediction - self.target) / self.target.size


class MLP(Module):
    """A classic feed-forward ANN with explicit forward and backward passes."""

    def __init__(self, config: MLPConfig) -> None:
        self.config = config
        dims = (config.input_dim, *config.hidden_dims, config.output_dim)
        self.layers = [
            Linear(dims[i], dims[i + 1], name=f"mlp.layers.{i}") for i in range(len(dims) - 1)
        ]
        self.activations = [make_activation(config.activation) for _ in config.hidden_dims]
        self.output_activation = (
            make_activation(config.output_activation) if config.output_activation is not None else None
        )
        self.last_activations: list[Array] = []

    def forward(self, x: Array) -> Array:
        self.last_activations = [x]
        for layer, activation in zip(self.layers[:-1], self.activations):
            x = activation.forward(layer.forward(x))
            self.last_activations.append(x)
        x = self.layers[-1].forward(x)
        if self.output_activation is not None:
            x = self.output_activation.forward(x)
        self.last_activations.append(x)
        return x

    def backward(self, grad: Array) -> Array:
        if self.output_activation is not None:
            grad = self.output_activation.backward(grad)
        grad = self.layers[-1].backward(grad)
        for layer, activation in reversed(list(zip(self.layers[:-1], self.activations))):
            grad = layer.backward(activation.backward(grad))
        return grad

    def parameters(self) -> list[Parameter]:
        params: list[Parameter] = []
        for layer in self.layers:
            params += layer.parameters()
        return params

    def parameter_count(self) -> int:
        return count_parameters(self.parameters())

    def neuron_debug(self, sample_index: int = 0) -> list[dict[str, object]]:
        debug: list[dict[str, object]] = []
        for layer_index, layer in enumerate(self.layers):
            if layer.x is None or layer.z is None:
                raise RuntimeError("MLP.neuron_debug called before forward")
            activation = self.last_activations[layer_index + 1]
            sample = min(sample_index, layer.x.shape[0] - 1)
            neurons = []
            for neuron_index in range(layer.weight.data.shape[1]):
                weights = layer.weight.data[:, neuron_index]
                inputs = layer.x[sample]
                neurons.append(
                    {
                        "index": neuron_index,
                        "weights": weights.copy(),
                        "bias": None if layer.bias is None else float(layer.bias.data[neuron_index]),
                        "input": inputs.copy(),
                        "contributions": inputs * weights,
                        "z": float(layer.z[sample, neuron_index]),
                        "activation": float(activation[sample, neuron_index]),
                        "gradZ": None
                        if layer.grad_output is None
                        else float(layer.grad_output[sample, neuron_index]),
                        "gradBias": None
                        if layer.bias is None
                        else float(layer.bias.grad[neuron_index]),
                        "gradWeights": layer.weight.grad[:, neuron_index].copy(),
                    }
                )
            debug.append(
                {
                    "index": layer_index,
                    "name": f"layer_{layer_index}",
                    "inputShape": layer.x.shape,
                    "outputShape": activation.shape,
                    "neurons": neurons,
                }
            )
        return debug
