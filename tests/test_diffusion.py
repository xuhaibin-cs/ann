import unittest

import numpy as np

from miniann_transformer.diffusion import DiffusionConfig, ToyDiffusion


class DiffusionTests(unittest.TestCase):
    def test_q_sample_and_snapshot_shapes(self) -> None:
        diffusion = ToyDiffusion(DiffusionConfig(image_size=8, timesteps=6, hidden_dim=16, batch_size=4), seed=9)
        x0 = diffusion.images[:2]
        noise = np.ones_like(x0)
        t = np.array([0, 5])
        noisy = diffusion.q_sample(x0, t, noise)
        self.assertEqual(noisy.shape, x0.shape)
        snapshot = diffusion.snapshot()
        self.assertEqual(snapshot["config"]["imageSize"], 8)
        self.assertEqual(len(snapshot["forward"]["clean"]), 8)
        self.assertEqual(len(snapshot["sample"]["frames"]), 4)

    def test_diffusion_training_updates_loss_history(self) -> None:
        diffusion = ToyDiffusion(DiffusionConfig(image_size=8, timesteps=6, hidden_dim=16, batch_size=4), seed=10)
        result = diffusion.train(2)
        self.assertEqual(result["trainedSteps"], 2)
        self.assertEqual(len(diffusion.loss_history), 2)
        self.assertGreater(result["losses"][0], 0.0)


if __name__ == "__main__":
    unittest.main()
