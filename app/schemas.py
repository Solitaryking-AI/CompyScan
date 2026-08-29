"""
Pydantic models describing the shape of ComplyScan's API responses.

Keeping these separate from business logic means the FastAPI route
signatures double as documentation (visible at /docs) without extra work.
"""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Axis-aligned box in pixel coordinates, top-left origin."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float


class TextBlock(BaseModel):
    """One raw text block as returned by the OCR engine, before classification."""
    text: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: BoundingBox


class DeclarationField(BaseModel):
    """
    One mandatory declaration (e.g. MRP, net quantity) after the classifier
    has matched it to a text block -- or determined it is missing.
    """
    field_name: str
    present: bool
    matched_text: Optional[str] = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    bbox: Optional[BoundingBox] = None
    notes: Optional[str] = None


class ComplianceResult(BaseModel):
    """Top-level response for a single scanned image."""
    filename: str
    image_width: int
    image_height: int
    raw_text_blocks: list[TextBlock]
    declarations: list[DeclarationField]
    missing_fields: list[str]
    compliant: bool
    warnings: list[str] = Field(default_factory=list)
