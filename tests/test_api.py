"""
Integration tests for the /scan endpoint.

We monkeypatch app.main.run_ocr instead of calling the real PaddleOCR
engine. This is deliberate, not a shortcut: PaddleOCR downloads its model
weights from a remote host on first use, which most CI runners and
sandboxed dev environments won't have open network access to. Mocking
here means these tests verify OUR code (upload validation, preprocessing
wiring, response shape, warning logic) runs correctly on every commit,
independent of whether the OCR model happens to be reachable.

Real end-to-end OCR accuracy should be checked separately, manually,
against real label photos -- see README "Testing with real images".
"""
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.main as main_module
from app.schemas import BoundingBox, TextBlock


def _fake_jpeg_bytes() -> bytes:
    img = Image.new("RGB", (200, 100), color="white")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def client():
    return TestClient(main_module.app)


def test_health_check(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_rejects_non_image_content_type(client):
    resp = client.post("/scan", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert resp.status_code == 415


def test_scan_with_all_fields_present(client, monkeypatch):
    def fake_run_ocr(image, lang="en"):
        bbox = BoundingBox(x_min=0, y_min=0, x_max=50, y_max=20)
        return [
            TextBlock(text="MRP Rs. 45.00 incl. of all taxes", confidence=0.95, bbox=bbox),
            TextBlock(text="Net Wt. 200g", confidence=0.92, bbox=bbox),
            TextBlock(text="Mfg. Date 03/2026", confidence=0.9, bbox=bbox),
            TextBlock(text="Consumer Care 1800-123-4567", confidence=0.88, bbox=bbox),
            TextBlock(text="Marketed by Acme Foods, PIN 400001", confidence=0.9, bbox=bbox),
        ]

    monkeypatch.setattr(main_module, "run_ocr", fake_run_ocr)

    resp = client.post("/scan", files={"file": ("label.jpg", _fake_jpeg_bytes(), "image/jpeg")})
    assert resp.status_code == 200
    data = resp.json()
    assert data["compliant"] is True
    assert data["missing_fields"] == []
    assert len(data["declarations"]) == 5


def test_scan_with_missing_mrp_is_flagged_not_hidden(client, monkeypatch):
    def fake_run_ocr(image, lang="en"):
        bbox = BoundingBox(x_min=0, y_min=0, x_max=50, y_max=20)
        return [TextBlock(text="Net Wt. 200g", confidence=0.9, bbox=bbox)]

    monkeypatch.setattr(main_module, "run_ocr", fake_run_ocr)

    resp = client.post("/scan", files={"file": ("label.jpg", _fake_jpeg_bytes(), "image/jpeg")})
    assert resp.status_code == 200
    data = resp.json()
    assert data["compliant"] is False
    assert "mrp" in data["missing_fields"]


def test_scan_with_no_text_detected_adds_warning(client, monkeypatch):
    monkeypatch.setattr(main_module, "run_ocr", lambda image, lang="en": [])

    resp = client.post("/scan", files={"file": ("blank.jpg", _fake_jpeg_bytes(), "image/jpeg")})
    assert resp.status_code == 200
    data = resp.json()
    assert any("no text detected" in w.lower() for w in data["warnings"])


def test_ocr_engine_failure_returns_502_not_500(client, monkeypatch):
    def broken_ocr(image, lang="en"):
        raise RuntimeError("model weights not found")

    monkeypatch.setattr(main_module, "run_ocr", broken_ocr)

    resp = client.post("/scan", files={"file": ("label.jpg", _fake_jpeg_bytes(), "image/jpeg")})
    assert resp.status_code == 502
