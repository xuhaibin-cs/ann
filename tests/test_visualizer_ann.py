import unittest

from visualizer.server import ClassicANNState


class ClassicANNVisualizerTests(unittest.TestCase):
    def test_snapshot_exposes_synchronized_math_trace(self) -> None:
        state = ClassicANNState(seed=107)
        snapshot = state.snapshot()
        trace = snapshot["mathTrace"]

        self.assertEqual(len(trace["samples"]), 4)
        self.assertEqual(len(trace["samples"][1]["layers"]), 3)
        self.assertAlmostEqual(trace["loss"], snapshot["loss"])
        self.assertGreater(abs(trace["samples"][1]["dLossDOutputZ"]), 0.0)
        self.assertGreater(snapshot["gradNorm"], 0.0)

    def test_training_records_real_adam_update(self) -> None:
        state = ClassicANNState(seed=107)
        result = state.train(1)
        update = result["ann"]["mathTrace"]["lastUpdate"]

        self.assertEqual(update["step"], 1)
        self.assertNotEqual(update["oldValue"], update["newValue"])
        self.assertAlmostEqual(update["newValue"] - update["oldValue"], update["delta"])


if __name__ == "__main__":
    unittest.main()
