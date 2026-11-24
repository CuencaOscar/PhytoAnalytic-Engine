"""Inference wrapper around the trained ResNet9 model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image

INPUT_SIZE = 256


@dataclass(frozen=True)
class Prediction:
    label: str
    probability: float  # percentage, 0-100


class Predictor(Protocol):
    def predict(self, image: Image.Image, top_k: int = 3) -> list[Prediction]: ...


class TorchPredictor:
    """Loads the weights once and serves top-k predictions for PIL images."""

    def __init__(self, weights_path: str | Path, device: str | None = None) -> None:
        import torch
        from torchvision import transforms

        from phytoanalytic.model.classes import CLASS_NAMES
        from phytoanalytic.model.resnet9 import ResNet9

        weights_path = Path(weights_path)
        if not weights_path.is_file():
            raise FileNotFoundError(
                f"Model weights not found at {weights_path}. Run `python scripts/download_model.py`."
            )

        self._torch = torch
        self._classes = CLASS_NAMES
        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._model = ResNet9(3, len(CLASS_NAMES)).to(self._device)
        self._model.load_state_dict(torch.load(weights_path, map_location=self._device))
        self._model.eval()
        self._transform = transforms.Compose([transforms.Resize((INPUT_SIZE, INPUT_SIZE)), transforms.ToTensor()])

    def predict(self, image: Image.Image, top_k: int = 3) -> list[Prediction]:
        torch = self._torch
        batch = self._transform(image.convert("RGB")).unsqueeze(0).to(self._device)
        with torch.no_grad():
            probs = torch.softmax(self._model(batch), dim=1)[0]
        top_probs, top_idx = torch.topk(probs, k=min(top_k, len(self._classes)))
        return [
            Prediction(label=self._classes[i], probability=float(p) * 100)
            for p, i in zip(top_probs.tolist(), top_idx.tolist(), strict=True)
        ]
