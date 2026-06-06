from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mlp import MLP, MLPConfig, MeanSquaredError
from .optim import Adam
from .tensor import Array


@dataclass
class DiffusionConfig:
    image_size: int = 16
    timesteps: int = 20
    beta_start: float = 5e-4
    beta_end: float = 8e-2
    hidden_dim: int = 128
    batch_size: int = 32


class ToyDiffusion:
    """A tiny DDPM-style denoising experiment for synthetic 16x16 images."""

    def __init__(self, config: DiffusionConfig | None = None, seed: int = 17) -> None:
        self.config = config or DiffusionConfig()
        self.rng = np.random.default_rng(seed)
        dim = self.config.image_size * self.config.image_size
        self.betas = np.linspace(self.config.beta_start, self.config.beta_end, self.config.timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = np.cumprod(self.alphas)
        self.images = make_shape_images(self.config.image_size)
        self.model = MLP(
            MLPConfig(
                input_dim=dim + 1,
                hidden_dims=(self.config.hidden_dim, self.config.hidden_dim),
                output_dim=dim,
                activation="tanh",
                output_activation=None,
            )
        )
        self.loss_fn = MeanSquaredError()
        self.optimizer = Adam(self.model.parameters(), lr=1e-3)
        self.step = 0
        self.loss_history: list[float] = []
        self.last_sample: dict[str, object] | None = None

    def q_sample(self, x0: Array, t: Array, noise: Array) -> Array:
        alpha_bar = self.alpha_bars[t][:, None]
        return np.sqrt(alpha_bar) * x0 + np.sqrt(1.0 - alpha_bar) * noise

    def model_input(self, noisy: Array, t: Array) -> Array:
        t_scaled = (t / max(self.config.timesteps - 1, 1)).astype(np.float64)[:, None]
        return np.concatenate([noisy, t_scaled], axis=-1)

    def train(self, steps: int) -> dict[str, object]:
        steps = max(1, min(steps, 500))
        losses = []
        for _ in range(steps):
            idx = self.rng.integers(0, len(self.images), size=self.config.batch_size)
            x0 = self.images[idx]
            t = self.rng.integers(0, self.config.timesteps, size=self.config.batch_size)
            noise = self.rng.normal(size=x0.shape)
            noisy = self.q_sample(x0, t, noise)
            predicted_noise = self.model.forward(self.model_input(noisy, t))
            loss = self.loss_fn.forward(predicted_noise, noise)
            self.optimizer.zero_grad()
            self.model.backward(self.loss_fn.backward())
            self.optimizer.step()
            self.step += 1
            losses.append(float(loss))
            self.loss_history.append(float(loss))
        return {"trainedSteps": steps, "losses": losses, "diffusion": self.snapshot()}

    def snapshot(self) -> dict[str, object]:
        forward = self.forward_demo()
        if self.last_sample is None:
            self.last_sample = self.sample()
        return {
            "config": {
                "imageSize": self.config.image_size,
                "timesteps": self.config.timesteps,
                "batchSize": self.config.batch_size,
                "hiddenDim": self.config.hidden_dim,
            },
            "step": self.step,
            "lossHistory": self.loss_history[-120:],
            "parameterCount": self.model.parameter_count(),
            "forward": forward,
            "sample": self.last_sample,
        }

    def forward_demo(self) -> dict[str, object]:
        clean = self.images[0:1]
        noise = np.linspace(-1.0, 1.0, clean.size, dtype=np.float64).reshape(clean.shape)
        steps = sorted({0, self.config.timesteps // 4, self.config.timesteps // 2, self.config.timesteps - 1})
        images = []
        for t in steps:
            noisy = self.q_sample(clean, np.array([t]), noise)[0]
            images.append({"t": int(t), "image": to_pixels(noisy, self.config.image_size)})
        return {"clean": to_pixels(clean[0], self.config.image_size), "noisy": images}

    def sample(self) -> dict[str, object]:
        dim = self.config.image_size * self.config.image_size
        x = self.rng.normal(size=(1, dim))
        keep = {self.config.timesteps - 1, (self.config.timesteps * 2) // 3, self.config.timesteps // 3, 0}
        frames = []
        for t in reversed(range(self.config.timesteps)):
            tt = np.array([t])
            predicted_noise = self.model.forward(self.model_input(x, tt))
            beta = self.betas[t]
            alpha = self.alphas[t]
            alpha_bar = self.alpha_bars[t]
            x = (x - beta * predicted_noise / np.sqrt(1.0 - alpha_bar)) / np.sqrt(alpha)
            if t > 0:
                x = x + np.sqrt(beta) * self.rng.normal(size=x.shape)
            if t in keep:
                frames.append({"t": int(t), "image": to_pixels(x[0], self.config.image_size)})
        self.last_sample = {"frames": frames, "final": frames[-1]["image"]}
        return self.last_sample


def make_shape_images(size: int) -> Array:
    images = []
    yy, xx = np.mgrid[0:size, 0:size]
    centers = [(size * 0.35, size * 0.35), (size * 0.5, size * 0.5), (size * 0.65, size * 0.45)]
    for cy, cx in centers:
        circle = ((yy - cy) ** 2 + (xx - cx) ** 2) < (size * 0.22) ** 2
        images.append(circle.astype(np.float64))
    for offset in [2, 4, 6]:
        square = np.zeros((size, size), dtype=np.float64)
        square[offset : offset + 7, offset : offset + 7] = 1.0
        images.append(square)
    for thickness in [1, 2, 3]:
        diagonal = np.abs(xx - yy) <= thickness
        images.append(diagonal.astype(np.float64))
        cross = (np.abs(xx - size // 2) <= thickness) | (np.abs(yy - size // 2) <= thickness)
        images.append(cross.astype(np.float64))
    arr = np.stack(images).reshape(len(images), -1)
    return arr * 2.0 - 1.0


def to_pixels(image: Array, size: int) -> list[list[float]]:
    pixels = np.clip((image.reshape(size, size) + 1.0) / 2.0, 0.0, 1.0)
    return pixels.tolist()
