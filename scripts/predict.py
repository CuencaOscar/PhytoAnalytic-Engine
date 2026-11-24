"""Classify a local image from the command line: `python scripts/predict.py leaf.jpg`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phytoanalytic.config import Settings  # noqa: E402
from phytoanalytic.predictor import TorchPredictor  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    predictor = TorchPredictor(Settings().model_path)
    for p in predictor.predict(Image.open(args.image), top_k=args.top_k):
        print(f"{p.label:<55} {p.probability:6.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
