"""
Maps raw OCR text blocks to Legal Metrology mandatory declaration fields.

This is the "no fixed template" solve described in the pitch: rather than
assuming MRP is always top-right and net quantity is always bottom-left,
each text block is scored against a set of regex/keyword patterns per
field. The highest-scoring block above a minimum confidence threshold wins
that field.

Phase 1 scope: five core declarations required on (almost) every packaged
commodity under Rule 6. Category-specific extra declarations and
exemptions belong to the rule engine (Phase 2), not here -- this module
only answers "did we find text that looks like X", not "was X required
for this product category".
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.schemas import DeclarationField, TextBlock

# Minimum OCR confidence before we're willing to trust a match at all.
# Low-confidence OCR reads (e.g. "MRQ" misread as "MRP") are worse than
# an honest "not found" -- false positives erode trust in the whole tool.
MIN_OCR_CONFIDENCE = 0.55


@dataclass
class FieldPattern:
    field_name: str
    label: str
    # Patterns are tried in order; first match wins the "matched_text" value,
    # but ANY pattern matching is enough to claim the block for this field.
    patterns: list[re.Pattern]


FIELD_PATTERNS: list[FieldPattern] = [
    FieldPattern(
        field_name="mrp",
        label="Maximum Retail Price (MRP)",
        patterns=[
            re.compile(r"\bMRP\b", re.IGNORECASE),
            re.compile(r"₹\s*\d+[.,]?\d*"),
            re.compile(r"\bRs\.?\s*\d+[.,]?\d*", re.IGNORECASE),
            re.compile(r"inclusive\s+of\s+all\s+taxes", re.IGNORECASE),
        ],
    ),
    FieldPattern(
        field_name="net_quantity",
        label="Net Quantity",
        patterns=[
            re.compile(r"\bnet\s*(qty|quantity|wt|weight)\b", re.IGNORECASE),
            re.compile(r"\d+\s*(g|kg|ml|l|ltr|gm|gms)\b", re.IGNORECASE),
            re.compile(r"निवल\s*मात्रा"),
        ],
    ),
    FieldPattern(
        field_name="mfg_date",
        label="Month & Year of Manufacture/Packing/Import",
        patterns=[
            re.compile(r"\b(mfg|mfd|manufactured|packed|pkd)\b", re.IGNORECASE),
            re.compile(r"\b(0[1-9]|1[0-2])\s*[/\-]\s*(19|20)\d{2}\b"),  # MM/YYYY
            re.compile(
                r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(19|20)\d{2}\b",
                re.IGNORECASE,
            ),
        ],
    ),
    FieldPattern(
        field_name="consumer_care",
        label="Consumer Care / Customer Support Details",
        patterns=[
            re.compile(r"\bconsumer\s*care\b", re.IGNORECASE),
            re.compile(r"\bcustomer\s*(care|support)\b", re.IGNORECASE),
            re.compile(r"\b[6-9]\d{9}\b"),  # 10-digit Indian mobile number
            re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),  # email
        ],
    ),
    FieldPattern(
        field_name="manufacturer_address",
        label="Name & Address of Manufacturer/Packer/Importer",
        patterns=[
            re.compile(r"\b(manufactured|marketed|packed|imported)\s+by\b", re.IGNORECASE),
            re.compile(r"\baddress\b", re.IGNORECASE),
            re.compile(r"\bpin\s*[:\-]?\s*\d{6}\b", re.IGNORECASE),  # Indian PIN code
        ],
    ),
]


def _score_block_for_field(text: str, pattern: FieldPattern) -> bool:
    return any(p.search(text) for p in pattern.patterns)


def classify_fields(blocks: list[TextBlock]) -> list[DeclarationField]:
    """
    For each mandatory field, find the best-matching text block (if any).

    A block is eligible to fill multiple fields if its text legitimately
    matches multiple patterns (e.g. an address block containing a PIN code
    AND being near a phone number) -- we don't remove matched blocks from
    the pool, since real labels often group related declarations together
    and a single OCR line can legitimately serve as evidence for more than
    one field.
    """
    results: list[DeclarationField] = []

    for field in FIELD_PATTERNS:
        best_block: TextBlock | None = None

        for block in blocks:
            if block.confidence < MIN_OCR_CONFIDENCE:
                continue
            if not _score_block_for_field(block.text, field):
                continue
            if best_block is None or block.confidence > best_block.confidence:
                best_block = block

        if best_block is not None:
            results.append(DeclarationField(
                field_name=field.field_name,
                present=True,
                matched_text=best_block.text,
                confidence=best_block.confidence,
                bbox=best_block.bbox,
            ))
        else:
            results.append(DeclarationField(
                field_name=field.field_name,
                present=False,
                notes=f"No text block matched patterns for '{field.label}'. "
                      f"Could be genuinely missing, or OCR/preprocessing failure "
                      f"-- verify manually before flagging as a violation.",
            ))

    return results


def missing_field_names(declarations: list[DeclarationField]) -> list[str]:
    return [d.field_name for d in declarations if not d.present]
