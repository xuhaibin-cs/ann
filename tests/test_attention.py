import unittest

import numpy as np

from miniann_transformer.attention import MultiHeadSelfAttention, ScaledDotProductAttention, causal_mask


class AttentionTests(unittest.TestCase):
    def test_causal_mask(self) -> None:
        mask = causal_mask(4)
        expected = np.array(
            [
                [True, False, False, False],
                [True, True, False, False],
                [True, True, True, False],
                [True, True, True, True],
            ]
        )
        np.testing.assert_array_equal(mask, expected)

    def test_scaled_attention_respects_mask(self) -> None:
        np.random.seed(0)
        attn = ScaledDotProductAttention()
        q = np.random.randn(2, 1, 3, 4)
        k = np.random.randn(2, 1, 3, 4)
        v = np.random.randn(2, 1, 3, 4)
        out, weights = attn.forward(q, k, v, causal_mask(3))
        self.assertEqual(out.shape, q.shape)
        np.testing.assert_allclose(weights[:, :, 0, 1:], 0.0, atol=1e-8)
        dq, dk, dv = attn.backward(np.ones_like(out))
        self.assertEqual(dq.shape, q.shape)
        self.assertEqual(dk.shape, k.shape)
        self.assertEqual(dv.shape, v.shape)

    def test_multi_head_self_attention_shapes(self) -> None:
        np.random.seed(1)
        mha = MultiHeadSelfAttention(embed_dim=8, num_heads=2)
        x = np.random.randn(3, 5, 8)
        y = mha.forward(x, causal_mask(5))
        dx = mha.backward(np.ones_like(y))
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(dx.shape, x.shape)
        self.assertEqual(mha.last_attention_weights.shape, (3, 2, 5, 5))


if __name__ == "__main__":
    unittest.main()
