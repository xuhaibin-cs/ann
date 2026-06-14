from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

from miniann_transformer.activations import softmax
from miniann_transformer.data import TOY_CORPUS, make_next_token_batch
from miniann_transformer.diffusion import ToyDiffusion
from miniann_transformer.generate import sample_next_token
from miniann_transformer.losses import CrossEntropyLoss
from miniann_transformer.mlp import MLP, MLPConfig, MeanSquaredError
from miniann_transformer.optim import Adam
from miniann_transformer.tokenizer import CharTokenizer
from miniann_transformer.transformer import TinyGPT, TransformerConfig


ROOT = Path(__file__).resolve().parent
STATIC_ROOT = ROOT / "static"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


class VisualizerState:
    def __init__(self, seed: int = 7) -> None:
        self.rng = np.random.default_rng(seed)
        np.random.seed(seed)
        self.tokenizer = CharTokenizer(TOY_CORPUS)
        self.encoded = self.tokenizer.encode(TOY_CORPUS)
        self.config = TransformerConfig(
            vocab_size=self.tokenizer.vocab_size,
            context_length=24,
            embed_dim=32,
            num_heads=4,
            num_layers=1,
            ff_dim=64,
            batch_size=8,
        )
        self.model = TinyGPT(self.config)
        self.loss_fn = CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=2e-3)
        self.step = 0
        self.loss_history: list[float] = []
        self.latest_sample = ""
        self.ann = ClassicANNState(seed=seed + 100)
        self.diffusion = ToyDiffusion(seed=seed + 200)

    def snapshot(self) -> dict[str, Any]:
        return {
            "config": self.config.__dict__,
            "parameterCount": self.model.parameter_count(),
            "vocab": [self.tokenizer.itos[i] for i in range(self.tokenizer.vocab_size)],
            "step": self.step,
            "lossHistory": self.loss_history[-120:],
            "latestSample": self.latest_sample,
            "paramNorm": self.param_norm(),
            "gradNorm": self.grad_norm(),
            "ann": self.ann.snapshot(),
            "diffusion": self.diffusion.snapshot(),
        }

    def forward(self, prompt: str) -> dict[str, Any]:
        ids = self._encode_prompt(prompt)
        context = ids[-self.config.context_length :]
        x = np.array([context], dtype=np.int64)
        logits, debug = self.model.forward(x, return_debug=True)
        probs = softmax(logits[0, -1], axis=-1)
        top_idx = np.argsort(probs)[::-1][:8]
        return {
            "prompt": prompt,
            "tokens": [self.tokenizer.itos[i] for i in context],
            "ids": context,
            "shapes": debug.shapes,
            "causalMask": debug.causal_mask.astype(int),
            "attentions": debug.attention_matrices,
            "topTokens": [
                {"token": self.tokenizer.itos[int(i)], "probability": float(probs[int(i)])}
                for i in top_idx
            ],
            "nextToken": self.tokenizer.itos[int(top_idx[0])],
            "snapshot": self.snapshot(),
        }

    def generate(self, prompt: str, max_new_tokens: int, temperature: float) -> dict[str, Any]:
        ids = self._encode_prompt(prompt)
        generated: list[dict[str, Any]] = []
        for _ in range(max(0, min(max_new_tokens, 160))):
            context = ids[-self.config.context_length :]
            x = np.array([context], dtype=np.int64)
            logits = self.model.forward(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            next_id = sample_next_token(logits[0, -1], temperature, rng=self.rng)
            ids.append(next_id)
            generated.append({"id": next_id, "token": self.tokenizer.itos[next_id]})
        text = self.tokenizer.decode(ids)
        self.latest_sample = text
        return {
            "text": text,
            "generated": generated,
            "forward": self.forward(self.tokenizer.decode(ids[-self.config.context_length :])),
        }

    def train(self, steps: int) -> dict[str, Any]:
        steps = max(1, min(steps, 200))
        losses = []
        for _ in range(steps):
            x, y = make_next_token_batch(
                self.encoded,
                self.config.batch_size,
                self.config.context_length,
                rng=self.rng,
            )
            logits = self.model.forward(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            loss = self.loss_fn.forward(logits, y)
            self.optimizer.zero_grad()
            self.model.backward(self.loss_fn.backward())
            self.optimizer.step()
            self.step += 1
            losses.append(float(loss))
            self.loss_history.append(float(loss))
        sample = self.generate("to ", max_new_tokens=64, temperature=0.8)["text"]
        return {
            "trainedSteps": steps,
            "losses": losses,
            "sample": sample,
            "snapshot": self.snapshot(),
            "forward": self.forward("to "),
        }

    def param_norm(self) -> float:
        total = sum(float(np.sum(param.data * param.data)) for param in self.model.parameters())
        return float(np.sqrt(total))

    def grad_norm(self) -> float:
        total = sum(float(np.sum(param.grad * param.grad)) for param in self.model.parameters())
        return float(np.sqrt(total))

    def _encode_prompt(self, prompt: str) -> list[int]:
        if not prompt:
            prompt = "to "
        unknown = sorted({ch for ch in prompt if ch not in self.tokenizer.stoi})
        if unknown:
            allowed = "".join(self.tokenizer.itos[i] for i in range(self.tokenizer.vocab_size))
            raise ValueError(f"Unknown character(s): {unknown}. Allowed characters: {allowed!r}")
        return self.tokenizer.encode(prompt)


class ClassicANNState:
    def __init__(self, seed: int = 107) -> None:
        self.rng = np.random.default_rng(seed)
        np.random.seed(seed)
        self.x = np.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.y = np.array([[0.0], [1.0], [1.0], [0.0]], dtype=np.float64)
        self.config = MLPConfig(input_dim=2, hidden_dims=(8, 8), output_dim=1, activation="tanh")
        self.model = MLP(self.config)
        self.loss_fn = MeanSquaredError()
        self.optimizer = Adam(self.model.parameters(), lr=5e-2)
        self.step = 0
        self.loss_history: list[float] = []
        self.last_update: dict[str, Any] | None = None

    def snapshot(self) -> dict[str, Any]:
        prediction = self.model.forward(self.x)
        loss = self.loss_fn.forward(prediction, self.y)
        self.optimizer.zero_grad()
        self.model.backward(self.loss_fn.backward())
        activations = self.activation_summary()
        neurons = self.model.neuron_debug(sample_index=0)
        math_trace = self.math_trace(prediction, loss)
        decision = self.decision_grid()
        return {
            "config": {
                "inputDim": self.config.input_dim,
                "hiddenDims": list(self.config.hidden_dims),
                "outputDim": self.config.output_dim,
                "activation": self.config.activation,
                "outputActivation": self.config.output_activation,
            },
            "step": self.step,
            "loss": loss,
            "lossHistory": self.loss_history[-120:],
            "parameterCount": self.model.parameter_count(),
            "paramNorm": self.param_norm(),
            "gradNorm": self.grad_norm(),
            "points": [
                {"x": float(point[0]), "y": float(point[1]), "label": int(label[0])}
                for point, label in zip(self.x, self.y)
            ],
            "predictions": [float(v) for v in prediction[:, 0]],
            "activations": activations,
            "neurons": neurons,
            "mathTrace": math_trace,
            "decision": decision,
        }

    def train(self, steps: int) -> dict[str, Any]:
        steps = max(1, min(steps, 1000))
        losses = []
        for _ in range(steps):
            prediction = self.model.forward(self.x)
            loss = self.loss_fn.forward(prediction, self.y)
            self.optimizer.zero_grad()
            self.model.backward(self.loss_fn.backward())
            params = self.model.parameters()
            before = [param.data.copy() for param in params]
            self.optimizer.step()
            self.step += 1
            first_param = params[0]
            m_hat = self.optimizer.m[0] / (1.0 - self.optimizer.beta1**self.optimizer.t)
            v_hat = self.optimizer.v[0] / (1.0 - self.optimizer.beta2**self.optimizer.t)
            self.last_update = {
                "step": self.step,
                "parameter": first_param.name,
                "index": [0, 0],
                "oldValue": float(before[0][0, 0]),
                "gradient": float(first_param.grad[0, 0]),
                "firstMoment": float(m_hat[0, 0]),
                "secondMoment": float(v_hat[0, 0]),
                "delta": float(first_param.data[0, 0] - before[0][0, 0]),
                "newValue": float(first_param.data[0, 0]),
            }
            losses.append(float(loss))
            self.loss_history.append(float(loss))
        return {"trainedSteps": steps, "losses": losses, "ann": self.snapshot()}

    def math_trace(self, prediction: np.ndarray, loss: float) -> dict[str, Any]:
        sample_traces = []
        activation_names = [self.config.activation] * len(self.config.hidden_dims) + [
            self.config.output_activation or "identity"
        ]
        for sample_index, (point, target) in enumerate(zip(self.x, self.y)):
            layers = []
            for layer_index, (layer, activation_name) in enumerate(zip(self.model.layers, activation_names)):
                if layer.x is None or layer.z is None or layer.grad_output is None:
                    raise RuntimeError("math trace requested before forward/backward")
                activation = self.model.last_activations[layer_index + 1]
                layer_input = layer.x[sample_index]
                contributions = layer_input[:, None] * layer.weight.data
                layers.append(
                    {
                        "index": layer_index,
                        "input": layer_input.copy(),
                        "inputShape": layer_input.shape,
                        "weights": layer.weight.data.copy(),
                        "bias": None if layer.bias is None else layer.bias.data.copy(),
                        "contributions": contributions,
                        "z": layer.z[sample_index].copy(),
                        "activation": activation[sample_index].copy(),
                        "activationName": activation_name,
                        "gradZ": layer.grad_output[sample_index].copy(),
                        "weightGrad": layer.weight.grad.copy(),
                        "biasGrad": None if layer.bias is None else layer.bias.grad.copy(),
                    }
                )
            pred = float(prediction[sample_index, 0])
            label = float(target[0])
            dloss_dprediction = float(2.0 * (pred - label) / self.y.size)
            sigmoid_derivative = pred * (1.0 - pred)
            sample_traces.append(
                {
                    "index": sample_index,
                    "input": point.copy(),
                    "target": label,
                    "prediction": pred,
                    "sampleSquaredError": float((pred - label) ** 2),
                    "dLossDPrediction": dloss_dprediction,
                    "outputActivationDerivative": sigmoid_derivative,
                    "dLossDOutputZ": dloss_dprediction * sigmoid_derivative,
                    "layers": layers,
                }
            )
        return {
            "objective": "XOR binary classification",
            "batchSize": int(self.x.shape[0]),
            "loss": float(loss),
            "lossFormula": "L = (1/N) sum_i (y_hat_i - y_i)^2",
            "optimizer": {
                "name": "Adam",
                "learningRate": self.optimizer.lr,
                "beta1": self.optimizer.beta1,
                "beta2": self.optimizer.beta2,
                "epsilon": self.optimizer.eps,
            },
            "samples": sample_traces,
            "lastUpdate": self.last_update,
        }

    def decision_grid(self, size: int = 48) -> dict[str, Any]:
        xs = np.linspace(-0.35, 1.35, size)
        ys = np.linspace(-0.35, 1.35, size)
        coords = np.array([[x, y] for y in ys for x in xs], dtype=np.float64)
        probs = self.model.forward(coords).reshape(size, size)
        return {
            "size": size,
            "xMin": float(xs[0]),
            "xMax": float(xs[-1]),
            "yMin": float(ys[0]),
            "yMax": float(ys[-1]),
            "values": probs,
        }

    def activation_summary(self) -> list[dict[str, Any]]:
        summary = []
        names = ["input", "hidden_0", "hidden_1", "output"]
        for name, values in zip(names, self.model.last_activations):
            summary.append(
                {
                    "name": name,
                    "shape": values.shape,
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }
            )
        return summary

    def param_norm(self) -> float:
        total = sum(float(np.sum(param.data * param.data)) for param in self.model.parameters())
        return float(np.sqrt(total))

    def grad_norm(self) -> float:
        total = sum(float(np.sum(param.grad * param.grad)) for param in self.model.parameters())
        return float(np.sqrt(total))


STATE = VisualizerState()


class VisualizerHandler(BaseHTTPRequestHandler):
    server_version = "MiniANNVisualizer/0.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/state":
            prompt = STATE.latest_sample or "to "
            self._send_json({"snapshot": STATE.snapshot(), "forward": STATE.forward(prompt)})
            return
        if parsed.path == "/":
            self._send_file(STATIC_ROOT / "index.html", "text/html; charset=utf-8")
            return
        path = (STATIC_ROOT / parsed.path.lstrip("/")).resolve()
        if not str(path).startswith(str(STATIC_ROOT.resolve())) or not path.exists():
            self.send_error(404)
            return
        self._send_file(path, self._content_type(path))

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
            parsed = urlparse(self.path)
            if parsed.path == "/api/forward":
                self._send_json(STATE.forward(str(payload.get("prompt", "to "))))
            elif parsed.path == "/api/generate":
                self._send_json(
                    STATE.generate(
                        str(payload.get("prompt", "to ")),
                        int(payload.get("maxNewTokens", 48)),
                        float(payload.get("temperature", 0.8)),
                    )
                )
            elif parsed.path == "/api/train":
                self._send_json(STATE.train(int(payload.get("steps", 10))))
            elif parsed.path == "/api/ann/train":
                self._send_json(STATE.ann.train(int(payload.get("steps", 50))))
            elif parsed.path == "/api/diffusion/train":
                self._send_json(STATE.diffusion.train(int(payload.get("steps", 20))))
            elif parsed.path == "/api/diffusion/sample":
                self._send_json({"diffusion": STATE.diffusion.snapshot() | {"sample": STATE.diffusion.sample()}})
            else:
                self.send_error(404)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=400)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(_jsonable(payload)).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _content_type(self, path: Path) -> str:
        suffix = path.suffix.lower()
        return {
            ".css": "text/css; charset=utf-8",
            ".js": "text/javascript; charset=utf-8",
            ".html": "text/html; charset=utf-8",
        }.get(suffix, "application/octet-stream")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MiniANN visualizer.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), VisualizerHandler)
    print(f"MiniANN visualizer running at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
