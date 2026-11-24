"""PhytoAnalytic Engine: REST API that classifies plant leaf diseases from a photo."""

from __future__ import annotations

from flask import Flask, jsonify, request

from phytoanalytic.config import Settings
from phytoanalytic.images import ImageInputError, image_from_upload, image_from_url
from phytoanalytic.predictor import Predictor

__version__ = "1.0.0"


def create_app(settings: Settings | None = None, predictor: Predictor | None = None) -> Flask:
    """Application factory. Pass a `predictor` to inject a fake one in tests."""
    settings = settings or Settings()
    if predictor is None:
        from phytoanalytic.predictor import TorchPredictor

        predictor = TorchPredictor(settings.model_path)

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = settings.max_image_bytes

    @app.get("/")
    @app.get("/health")
    def health():
        return jsonify(status="ok", service="phytoanalytic-engine", version=__version__)

    @app.post("/api")
    def classify():
        """Accepts a multipart file `image`, or a form field `image` with an allowed HTTPS URL."""
        try:
            if "image" in request.files:
                image = image_from_upload(request.files["image"], settings.max_image_bytes)
            elif request.form.get("image"):
                image = image_from_url(
                    request.form["image"],
                    settings.allowed_image_hosts,
                    settings.max_image_bytes,
                    settings.download_timeout,
                )
            else:
                return jsonify(error="Send an image file or an image URL in the `image` field."), 400
        except ImageInputError as exc:
            return jsonify(error=str(exc)), 400

        predictions = predictor.predict(image, top_k=settings.top_k)
        # Response shape kept compatible with the LeafHealth Tracker mobile app.
        return jsonify([{"class": p.label, "probability": f"{p.probability:.2f}"} for p in predictions])

    return app
