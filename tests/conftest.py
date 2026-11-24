import io

import pytest
from PIL import Image

from phytoanalytic import create_app
from phytoanalytic.config import Settings
from phytoanalytic.predictor import Prediction


class FakePredictor:
    def __init__(self):
        self.calls = 0

    def predict(self, image, top_k=3):
        self.calls += 1
        return [Prediction("Tomato___healthy", 91.2345), Prediction("Tomato___Early_blight", 5.5)][:top_k]


@pytest.fixture
def predictor():
    return FakePredictor()


@pytest.fixture
def client(predictor):
    app = create_app(Settings(model_path="unused", allowed_image_hosts=("firebasestorage.googleapis.com",)), predictor)
    app.testing = True
    return app.test_client()


@pytest.fixture
def png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color=(30, 160, 60)).save(buf, format="PNG")
    return buf.getvalue()
