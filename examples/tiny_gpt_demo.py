import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from miniann_transformer.train import train_toy_model


if __name__ == "__main__":
    train_toy_model(steps=80, print_every=20)
