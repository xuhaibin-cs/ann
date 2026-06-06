import unittest

import numpy as np

from miniann_transformer.mlp import MLP, MLPConfig, MeanSquaredError
from miniann_transformer.optim import Adam


class MLPTests(unittest.TestCase):
    def test_mlp_forward_backward_shapes(self) -> None:
        np.random.seed(4)
        model = MLP(MLPConfig(input_dim=2, hidden_dims=(4, 3), output_dim=1, activation="tanh"))
        x = np.array([[0.0, 1.0], [1.0, 0.0]])
        y = np.array([[1.0], [1.0]])
        pred = model.forward(x)
        self.assertEqual(pred.shape, (2, 1))
        loss_fn = MeanSquaredError()
        loss = loss_fn.forward(pred, y)
        model.zero_grad()
        dx = model.backward(loss_fn.backward())
        self.assertGreater(loss, 0.0)
        self.assertEqual(dx.shape, x.shape)
        total_grad = sum(float(np.abs(param.grad).sum()) for param in model.parameters())
        self.assertGreater(total_grad, 0.0)
        debug = model.neuron_debug(sample_index=0)
        self.assertEqual(len(debug), 3)
        self.assertEqual(len(debug[0]["neurons"]), 4)
        first = debug[0]["neurons"][0]
        self.assertIn("weights", first)
        self.assertIn("contributions", first)
        self.assertIn("z", first)
        self.assertIn("activation", first)

    def test_mlp_training_step_updates_params(self) -> None:
        np.random.seed(5)
        model = MLP(MLPConfig(input_dim=2, hidden_dims=(6,), output_dim=1, activation="relu"))
        x = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        y = np.array([[0.0], [1.0], [1.0], [0.0]])
        loss_fn = MeanSquaredError()
        opt = Adam(model.parameters(), lr=1e-2)
        pred = model.forward(x)
        loss_fn.forward(pred, y)
        before = [p.data.copy() for p in model.parameters()]
        opt.zero_grad()
        model.backward(loss_fn.backward())
        opt.step()
        changed = any(not np.allclose(p.data, old) for p, old in zip(model.parameters(), before))
        self.assertTrue(changed)


if __name__ == "__main__":
    unittest.main()
