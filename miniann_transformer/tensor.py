from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


Array = np.ndarray


@dataclass
class Parameter:
    """A trainable NumPy array and its gradient."""

    data: Array
    grad: Array
    name: str = ""

    @classmethod
    def from_data(cls, data: Array, name: str = "") -> "Parameter":
        return cls(data=data.astype(np.float64), grad=np.zeros_like(data, dtype=np.float64), name=name)

    def zero_grad(self) -> None:
        self.grad.fill(0.0)


class Module:
    """Tiny base class for explicit forward/backward modules."""

    def parameters(self) -> list[Parameter]:
        return []

    def zero_grad(self) -> None:
        for param in self.parameters():
            param.zero_grad()

    def named_parameters(self) -> Iterator[tuple[str, Parameter]]:
        for param in self.parameters():
            yield param.name, param


def count_parameters(params: list[Parameter]) -> int:
    return int(sum(param.data.size for param in params))


def one_hot(ids: Array, num_classes: int) -> Array:
    out = np.zeros((*ids.shape, num_classes), dtype=np.float64)
    np.put_along_axis(out, ids[..., None], 1.0, axis=-1)
    return out
