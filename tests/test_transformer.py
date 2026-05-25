import unittest

import numpy as np

from miniann_transformer.losses import CrossEntropyLoss
from miniann_transformer.optim import Adam
from miniann_transformer.transformer import TinyGPT, TransformerConfig


class TransformerTests(unittest.TestCase):
    def test_tiny_gpt_forward_backward_and_debug(self) -> None:
        np.random.seed(2)
        config = TransformerConfig(
            vocab_size=12,
            context_length=6,
            embed_dim=8,
            num_heads=2,
            num_layers=1,
            ff_dim=16,
            batch_size=2,
        )
        model = TinyGPT(config)
        x = np.array([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]])
        logits, debug = model.forward(x, return_debug=True)
        self.assertEqual(logits.shape, (2, 6, 12))
        self.assertEqual(debug.shapes["logits"], (2, 6, 12))
        self.assertEqual(len(debug.attention_matrices), 1)
        self.assertGreater(debug.parameter_count, 0)
        y = np.array([[2, 3, 4, 5, 6, 7], [5, 4, 3, 2, 1, 0]])
        loss_fn = CrossEntropyLoss()
        loss = loss_fn.forward(logits, y)
        model.zero_grad()
        model.backward(loss_fn.backward())
        self.assertGreater(loss, 0.0)
        total_grad = sum(float(np.abs(param.grad).sum()) for param in model.parameters())
        self.assertGreater(total_grad, 0.0)

    def test_one_training_step_changes_loss_or_params(self) -> None:
        np.random.seed(3)
        config = TransformerConfig(vocab_size=8, context_length=4, embed_dim=8, num_heads=2, num_layers=1, ff_dim=16)
        model = TinyGPT(config)
        x = np.array([[0, 1, 2, 3], [3, 2, 1, 0]])
        y = np.array([[1, 2, 3, 4], [2, 1, 0, 1]])
        loss_fn = CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=1e-3)
        logits = model.forward(x)
        loss = loss_fn.forward(logits, y)
        before = [p.data.copy() for p in model.parameters()]
        optimizer.zero_grad()
        model.backward(loss_fn.backward())
        optimizer.step()
        changed = any(not np.allclose(p.data, old) for p, old in zip(model.parameters(), before))
        self.assertTrue(changed)
        self.assertGreater(loss, 0.0)


if __name__ == "__main__":
    unittest.main()
