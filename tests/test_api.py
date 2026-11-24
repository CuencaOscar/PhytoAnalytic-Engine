import io


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.get_json()["status"] == "ok"


def test_classify_upload_returns_top_predictions(client, predictor, png_bytes):
    res = client.post("/api", data={"image": (io.BytesIO(png_bytes), "leaf.png")}, content_type="multipart/form-data")
    assert res.status_code == 200
    body = res.get_json()
    assert body[0] == {"class": "Tomato___healthy", "probability": "91.23"}
    assert predictor.calls == 1


def test_missing_image_is_rejected(client):
    res = client.post("/api", data={})
    assert res.status_code == 400


def test_invalid_file_is_rejected(client):
    res = client.post(
        "/api", data={"image": (io.BytesIO(b"not an image"), "x.png")}, content_type="multipart/form-data"
    )
    assert res.status_code == 400


def test_url_from_disallowed_host_is_rejected(client, predictor):
    res = client.post("/api", data={"image": "https://evil.example.com/leaf.jpg"})
    assert res.status_code == 400
    assert predictor.calls == 0
