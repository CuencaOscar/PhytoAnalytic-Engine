"""Safe loading of images sent to the API, either as an upload or as an allowed URL."""

from __future__ import annotations

import io
from urllib.parse import urlparse

import requests
from PIL import Image, UnidentifiedImageError


class ImageInputError(ValueError):
    """Raised when the request does not contain a valid image."""


def _open_image(data: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageInputError("The file is not a valid image.") from exc
    return image


def image_from_upload(file_storage, max_bytes: int) -> Image.Image:
    data = file_storage.read(max_bytes + 1)
    if not data:
        raise ImageInputError("The uploaded file is empty.")
    if len(data) > max_bytes:
        raise ImageInputError("The image is too large.")
    return _open_image(data)


def is_allowed_url(url: str, allowed_hosts: tuple[str, ...]) -> bool:
    parsed = urlparse(url)
    return parsed.scheme == "https" and parsed.hostname in allowed_hosts


def image_from_url(url: str, allowed_hosts: tuple[str, ...], max_bytes: int, timeout: float) -> Image.Image:
    if not is_allowed_url(url, allowed_hosts):
        raise ImageInputError("Image URL host is not allowed.")
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ImageInputError("The image could not be downloaded.") from exc

    chunks, size = [], 0
    for chunk in response.iter_content(chunk_size=64 * 1024):
        size += len(chunk)
        if size > max_bytes:
            raise ImageInputError("The image is too large.")
        chunks.append(chunk)
    return _open_image(b"".join(chunks))
