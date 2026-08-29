"""
ComplyScan Phase 1 MVP -- OCR + mandatory-field extraction API.

Run locally:
    uvicorn app.main:app --reload --port 8000

Then POST an image to /scan (see README for a curl example), or open
/docs for the interactive Swagger UI.

Scope reminder: this phase answers "what mandatory fields did we find on
this label, and what's missing" -- it does NOT yet apply category-specific
rules/exemptions (Phase 2), measure font size in mm (Phase 3), or check
for MRP tampering (Phase 3/4). Those build on top of this extraction
output, they don't replace it.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.field_classifier import classify_fields, missing_field_names
from app.ocr_engine import run_ocr
from app.preprocessing import preprocess
from app.schemas import ComplianceResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ComplyScan",
    description="AI-based Legal Metrology label compliance scanner -- Phase 1 (OCR + field extraction)",
    version="0.1.0",
)

# Permissive CORS for local hackathon dev (React app on a different port).
# Tighten this before any real deployment -- see README "Before you deploy".
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/scan", response_model=ComplianceResult)
async def scan_label(file: UploadFile = File(...)) -> ComplianceResult:
    """
    Accepts one label image, runs preprocessing + OCR + field classification,
    and returns which mandatory declarations were found vs. missing.
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content type '{file.content_type}'. "
                   f"Allowed: {sorted(ALLOWED_CONTENT_TYPES)}",
        )

    raw_bytes = await file.read()
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image too large (max 15MB)")

    warnings: list[str] = []

    try:
        image = preprocess(raw_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        text_blocks = run_ocr(image)
    except Exception as exc:  # noqa: BLE001 -- OCR engine failures shouldn't 500 silently
        logger.exception("OCR engine failed")
        raise HTTPException(status_code=502, detail=f"OCR engine error: {exc}") from exc

    if not text_blocks:
        warnings.append(
            "No text detected at all. Check image quality/focus, or whether "
            "the label is genuinely blank/unreadable in this photo."
        )

    declarations = classify_fields(text_blocks)
    missing = missing_field_names(declarations)

    low_conf_blocks = [b for b in text_blocks if b.confidence < 0.5]
    if len(low_conf_blocks) > len(text_blocks) * 0.4 and text_blocks:
        warnings.append(
            "Over 40% of detected text has low OCR confidence -- results for "
            "this image should be treated as preliminary and manually verified."
        )

    height, width = image.shape[:2]

    return ComplianceResult(
        filename=file.filename or "unknown",
        image_width=width,
        image_height=height,
        raw_text_blocks=text_blocks,
        declarations=declarations,
        missing_fields=missing,
        compliant=(len(missing) == 0),
        warnings=warnings,
    )
