from __future__ import annotations

import numpy as np

from .tensor import Parameter


class SGD:
    def __init__(self, params: list[Parameter], lr: float = 1e-2) -> None:
        self.params = params
        self.lr = lr

    def step(self) -> None:
        for param in self.params:
            param.data -= self.lr * param.grad

    def zero_grad(self) -> None:
        for param in self.params:
            param.zero_grad()


class Adam:
    def __init__(
        self,
        params: list[Parameter],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(param.data) for param in params]
        self.v = [np.zeros_like(param.data) for param in params]

    def step(self) -> None:
        self.t += 1
        for i, param in enumerate(self.params):
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * param.grad
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (param.grad * param.grad)
            m_hat = self.m[i] / (1.0 - self.beta1**self.t)
            v_hat = self.v[i] / (1.0 - self.beta2**self.t)
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self) -> None:
        for param in self.params:
            param.zero_grad()
