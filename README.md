# MiniANN-Transformer

MiniANN-Transformer is a minimal decoder-only Transformer written for research and education. It uses Python and NumPy only, with explicit `forward` and `backward` methods instead of autograd.

This is not intended to replace PyTorch, TensorFlow, or JAX. The point is to make the path from ANN primitives to a tiny GPT-style language model readable.

## Quick Start

MiniANN-Transformer requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
```

Start the interactive browser lab:

```bash
python -m visualizer.server
```

Then open [http://127.0.0.1:8765](http://127.0.0.1:8765). The server uses only the Python standard library and NumPy; the frontend is served directly, with no JavaScript build step.

## Progressive Learning Path

This project is designed to grow from simple to complex. The early code favors small, explicit implementations that are easy to inspect and reason about. Later improvements can build on that foundation by adding richer training loops, better visualization, more model components, and eventually more realistic Transformer features.

In other words, the repository is not meant to appear fully polished all at once. It is a step-by-step learning project: start with the smallest working neural-network pieces, understand how they fit together, then keep refining the system toward more capable models.

## What Is Implemented

- ANN basics: `Linear`, ReLU, GELU, stable softmax, cross entropy, SGD, Adam
- Transformer pieces: token embeddings, positional embeddings, LayerNorm, causal masking, scaled dot-product attention, multi-head self-attention, feed-forward networks, residual blocks
- Tiny GPT model: embeddings, decoder blocks, final LayerNorm, and a language-model head
- Toy character-level training loop
- Autoregressive generation with temperature sampling
- Debug outputs: tensor shapes, causal mask, attention matrices, token probabilities hook points, and parameter count
- Browser lab with a numerical ANN training trace from input through parameter update

## Interactive Browser Lab

| View | What you can inspect |
| --- | --- |
| Classic ANN | XOR inputs, neuron activations, the decision boundary, MSE gradients, layer-by-layer backpropagation, and a real Adam parameter update |
| Transformer | Tokenization, tensor shapes, causal masks, attention matrices, next-token probabilities, training loss, and generated text |
| Diffusion | Synthetic inspection patterns, forward noising, denoiser training loss, and the reverse sampling trajectory |

Training and sampling happen in the local Python process. The browser calls small JSON endpoints exposed by `visualizer/server.py`, so every displayed value comes from the same NumPy implementation used by the tests and examples.

## Repository Guide

```text
miniann_transformer/
  tensor.py          Parameter and Module foundations
  layers.py          Linear and feed-forward layers
  activations.py     ReLU, GELU, Tanh, Sigmoid, and softmax
  attention.py       Causal attention and multi-head self-attention
  transformer.py     Decoder block and TinyGPT
  mlp.py             Classic feed-forward ANN
  diffusion.py       Toy DDPM-style denoising model
  optim.py           SGD and Adam
visualizer/           Local JSON server and browser interface
examples/             Runnable tiny GPT demo
tests/                Unit tests for ANN, Transformer, diffusion, and visualizer traces
docs/screenshots/     README images
```

## ANN Components To Transformer

A Transformer block is built from ordinary neural-network parts:

- `Linear` layers project embeddings into queries, keys, values, feed-forward hidden states, and vocabulary logits.
- GELU adds nonlinearity inside the feed-forward network.
- Softmax turns attention scores and output logits into probability distributions.
- LayerNorm stabilizes activations before attention and feed-forward sublayers.
- Residual connections preserve information and improve gradient flow.

The browser lab makes these pieces inspectable from the parameter level up to generated behavior.

![MiniANN Transformer Lab visualizer](docs/screenshots/visualizer-lab.png)

## Mathematical Foundations

### Classic ANN

The Classic ANN lab trains a `2 -> 8 -> 8 -> 1` multilayer perceptron on all four XOR examples. The two hidden layers use Tanh, the output uses Sigmoid, the loss is mean squared error, and the optimizer is Adam.

For one input vector $x \in \mathbb{R}^2$, the forward pass composes affine maps and elementwise nonlinearities:

$$
\begin{aligned}
z_1 &= x W_1 + b_1,        & h_1 &= \tanh(z_1), \\
z_2 &= h_1 W_2 + b_2,      & h_2 &= \tanh(z_2), \\
z_3 &= h_2 W_3 + b_3,      & \hat{y} &= \sigma(z_3), \\
L &= \frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i - y_i)^2.
\end{aligned}
$$

The output responsibility begins with the MSE and Sigmoid derivatives:

$$
\frac{\partial L}{\partial \hat{y}}
= \frac{2(\hat{y}-y)}{N},
\qquad
\frac{\partial \hat{y}}{\partial z_3}
= \hat{y}(1-\hat{y}),
\qquad
\delta_3
= \frac{\partial L}{\partial z_3}
= \frac{\partial L}{\partial \hat{y}}
  \frac{\partial \hat{y}}{\partial z_3}.
$$

Backpropagation applies the chain rule from right to left. For a linear layer followed by activation $f$:

$$
\frac{\partial L}{\partial W_l}
= a_{l-1}^{\mathsf T}\delta_l,
\qquad
\frac{\partial L}{\partial b_l}
= \sum \delta_l,
\qquad
\delta_{l-1}
= (\delta_l W_l^{\mathsf T}) \odot f'(z_{l-1}).
$$

Each module stores the intermediate values required by its explicit `backward` method. Adam then updates every parameter using first and second gradient moments:

$$
\begin{aligned}
m_t &= \beta_1m_{t-1} + (1-\beta_1)g_t, \\
v_t &= \beta_2v_{t-1} + (1-\beta_2)g_t^2, \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^t},
& \hat{v}_t &= \frac{v_t}{1-\beta_2^t}, \\
\theta_t &= \theta_{t-1}
- \eta\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}.
\end{aligned}
$$

The browser lab exposes this computation as a ten-step mathematical trace:

1. Input vector and target
2. Weighted sum $z=xW+b$
3. Nonlinear activation
4. Hidden representations
5. Sigmoid output prediction
6. MSE calculation
7. Chain-rule calculation at the output
8. Layer-by-layer gradient propagation
9. A real Adam parameter update with old value, update amount, and new value
10. Repetition with the updated parameters

The trace can switch among the four XOR samples. It shows formulas, tensor shapes, concrete numerical substitutions, layer gradient norms, predictions, and a synchronized decision boundary. XOR is intentionally low-dimensional so the nonlinear boundary is visible, while the current `2 -> 8 -> 8 -> 1` model remains large enough to inspect learned hidden representations.

### Decoder-Only Transformer

For a batch of integer token ids $x$ with shape $B \times T$, the model forms token and position embeddings:

$$
X = E_{\text{token}}[x] + E_{\text{pos}}[0:T],
\qquad X \in \mathbb{R}^{B \times T \times d_{\text{model}}}.
$$

Each attention head projects $X$ into queries, keys, and values with head dimension $d_{\text{head}} = d_{\text{model}} / n_{\text{heads}}$:

$$
\begin{aligned}
Q &= XW_Q + b_Q, &
K &= XW_K + b_K, &
V &= XW_V + b_V, \\
S &= \frac{QK^\top}{\sqrt{d_{\text{head}}}}, &
A &= \mathrm{softmax}(\mathrm{mask}(S)), &
H &= AV.
\end{aligned}
$$

The causal mask keeps entries with source position $j \le t$ and replaces future scores with a very negative number before softmax. This makes the row $A[t,:]$ a probability distribution over only the current and previous tokens.

Multi-head attention runs this calculation in parallel heads, concatenates the head outputs, and applies an output projection:

$$
\mathrm{MHA}(X) =
\mathrm{concat}(H_1,\ldots,H_n)W_O + b_O.
$$

The feed-forward sublayer is a position-wise MLP shared across time:

$$
\mathrm{FFN}(x) =
\mathrm{GELU}(xW_1 + b_1)W_2 + b_2.
$$

This implementation uses a pre-norm decoder block:

$$
\begin{aligned}
X_1 &= X + \mathrm{MHA}(\mathrm{LayerNorm}(X)), \\
X_2 &= X_1 + \mathrm{FFN}(\mathrm{LayerNorm}(X_1)).
\end{aligned}
$$

The final LayerNorm and language-model head produce logits $Z \in \mathbb{R}^{B \times T \times |V|}$. Training uses teacher forcing: the input sequence is paired with the same sequence shifted one token to the left. Cross entropy is averaged over all batch and time positions:

$$
\begin{aligned}
p &= \mathrm{softmax}(Z), \\
L &= -\frac{1}{BT}\sum_{b=1}^{B}\sum_{t=1}^{T}
\log p_{b,t,y_{b,t}}, \\
\frac{\partial L}{\partial Z}
&= \frac{p - \mathrm{onehot}(y)}{BT}.
\end{aligned}
$$

The derivative above is why the code can implement cross entropy and softmax together in a compact, numerically stable way.

![Transformer attention and causal mask](docs/screenshots/transformer-attention.png)

### Industrial-Style Toy Diffusion

The diffusion demo is framed as a minimal proxy for industrial visual inspection. In real pillar industries such as manufacturing, semiconductors, energy infrastructure, medical imaging, and remote sensing, models often need to recover useful structure from noisy or partially corrupted measurements. Here, the "parts" are synthetic geometric patterns rather than real wafers, welds, turbines, scans, or satellite tiles, but the learning problem is the same at a small scale: learn the clean signal distribution and remove noise step by step.

It uses a tiny DDPM-style denoising setup with a fixed variance schedule:

$$
\beta_t \text{ increases linearly from } \beta_{\text{start}}
\text{ to } \beta_{\text{end}}, \qquad
\alpha_t = 1 - \beta_t, \qquad
\bar{\alpha}_t = \prod_{s=0}^{t}\alpha_s.
$$

For a clean inspection image $x_0$, noise $\epsilon \sim \mathcal{N}(0,I)$, and timestep $t$, the closed-form forward noising process is:

$$
x_t = \sqrt{\bar{\alpha}_t}\,x_0
      {}+ \sqrt{1-\bar{\alpha}_t}\,\epsilon,
\qquad
\epsilon \sim \mathcal{N}(0,I).
$$

The denoiser receives the noisy image plus a scaled timestep, then predicts the injected noise. This mirrors the practical idea behind many industrial restoration and anomaly workflows: model what normal structure looks like, then use the reconstruction or denoising behavior to expose corruption, damage, or uncertainty.

$$
\hat{\epsilon} = \epsilon_{\theta}(x_t,t),
\qquad
L = \mathbb{E}\left[\frac{1}{D}\lVert \epsilon - \hat{\epsilon} \rVert_2^2\right].
$$

Sampling starts from Gaussian noise and applies the simplified noise-prediction reverse update used in this project. The extra noise term is used for $t > 0$ and omitted at the final step:

$$
x_{t-1} =
\frac{x_t - \beta_t \epsilon_{\theta}(x_t,t) / \sqrt{1-\bar{\alpha}_t}}{\sqrt{\alpha_t}}
      {}+ \sqrt{\beta_t}\,z,
\qquad
z \sim \mathcal{N}(0,I),\ t > 0.
$$

This is intentionally small and practical rather than a complete generative-model framework: there is no U-Net, learned variance, classifier-free guidance, advanced sampler, large dataset, or domain-specific sensor model. The value is that the forward process, denoising objective, and reverse trajectory are all visible in a few NumPy operations, while still pointing toward serious industrial uses such as defect inspection, image restoration, simulation-based data augmentation, and uncertainty-aware monitoring.

![Toy diffusion forward noising and reverse sampling](docs/screenshots/diffusion-process.png)

### Optimization

All gradients are computed by explicit module `backward` methods rather than autograd. Each parameter stores both data and gradient arrays. The training loops use SGD or Adam-style updates:

$$
\theta \leftarrow \theta - \eta \nabla_{\theta}L
$$

For Adam, the implementation keeps first and second moment estimates:

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1)g_t, \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)g_t^2, \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^t}, \\
\hat{v}_t &= \frac{v_t}{1-\beta_2^t}, \\
\theta_t &\leftarrow \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}.
\end{aligned}
$$

The lab exposes loss curves, parameter norms, gradient norms, attention matrices, masks, and sampled text so numerical training dynamics can be inspected alongside model behavior.

![Transformer training loss and sampled text](docs/screenshots/transformer-training.png)

## Why Decoder-Only

Decoder-only models predict the next token from previous tokens. That makes them ideal for a tiny educational GPT-style implementation: the training target is just the input shifted by one position, and causal masking prevents the model from seeing future tokens.

## How Attention Works

Scaled dot-product attention computes:

```text
scores = QK^T / sqrt(head_dim)
weights = softmax(mask(scores))
output = weights V
```

The causal mask keeps each position from attending to tokens on its right. Multi-head attention repeats this process in several smaller representation spaces, then concatenates the heads and applies a final linear projection.

## Train The Tiny Model

Run the toy demo:

```bash
python examples/tiny_gpt_demo.py
```

Or run the module directly:

```bash
python -m miniann_transformer.train
```

The demo trains on a tiny character-level corpus, prints loss, and then samples text.

## Generate Text

Use `generate` with a trained model and tokenizer:

```python
from miniann_transformer.generate import generate

text = generate(model, tokenizer, prompt="to ", max_new_tokens=80, temperature=0.8)
print(text)
```

Use `temperature=0.0` for greedy decoding.

## Debug And Visualization

The **Classic ANN** tab is organized around the full mathematical training chain described above. The Transformer and Diffusion tabs expose their corresponding internal states and training behavior. See [Quick Start](#quick-start) for the launch command.

Call:

```python
logits, debug = model.forward(token_ids, return_debug=True)
```

`debug` contains layer output shapes, the causal mask, per-layer attention matrices, token probabilities, and parameter count.

## Tests

Run:

```bash
python -m unittest discover -s tests -v
```

## Limitations

- This code prioritizes clarity over speed.
- Backpropagation is manual and only covers the modules included here.
- There is no batching pipeline beyond the toy helper.
- No GPU, mixed precision, dropout, checkpointing, or advanced sampling.
- The demo corpus is intentionally tiny, so generated text is for mechanics, not quality.
