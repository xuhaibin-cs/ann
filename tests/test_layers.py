import unittest

import numpy as np

from miniann_transformer.activations import GELU, ReLU, softmax
from miniann_transformer.layers import Linear
from miniann_transformer.losses import CrossEntropyLoss
from miniann_transformer.optim import SGD


class LayerTests(unittest.TestCase):
    def test_linear_backward_shapes_and_gradients(self) -> None:
        np.random.seed(0)
        layer = Linear(3, 2)
        x = np.random.randn(4, 5, 3)
        y = layer.forward(x)
        grad = np.ones_like(y)
        dx = layer.backward(grad)
        self.assertEqual(y.shape, (4, 5, 2))
        self.assertEqual(dx.shape, x.shape)
        self.assertEqual(layer.weight.grad.shape, (3, 2))
        self.assertEqual(layer.bias.grad.shape, (2,))

    def test_relu_and_gelu_backward(self) -> None:
        x = np.array([[-1.0, 0.0, 2.0]])
        relu = ReLU()
        np.testing.assert_allclose(relu.forward(x), [[0.0, 0.0, 2.0]])
        np.testing.assert_allclose(relu.backward(np.ones_like(x)), [[0.0, 0.0, 1.0]])
        gelu = GELU()
        y = gelu.forward(x)
        dx = gelu.backward(np.ones_like(x))
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(dx.shape, x.shape)

    def test_softmax_is_stable(self) -> None:
        x = np.array([[1000.0, 1001.0, 1002.0]])
        probs = softmax(x)
        self.assertFalse(np.isnan(probs).any())
        np.testing.assert_allclose(probs.sum(axis=-1), np.ones(1))

    def test_cross_entropy_and_sgd(self) -> None:
        logits = np.array([[[2.0, 0.0], [0.5, 1.5]]])
        targets = np.array([[0, 1]])
        loss_fn = CrossEntropyLoss()
        loss = loss_fn.forward(logits, targets)
        grad = loss_fn.backward()
        self.assertGreater(loss, 0.0)
        self.assertEqual(grad.shape, logits.shape)
        layer = Linear(2, 2)
        layer.weight.grad.fill(1.0)
        old = layer.weight.data.copy()
        SGD(layer.parameters(), lr=0.1).step()
        np.testing.assert_allclose(layer.weight.data, old - 0.1)


if __name__ == "__main__":
    unittest.main()
