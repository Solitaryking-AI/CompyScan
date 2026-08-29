"""
Unit tests for field_classifier.py.

These deliberately construct TextBlock objects by hand instead of running
real OCR -- the classifier's job is pure text-pattern logic, and testing
it this way means the suite runs in milliseconds with no model downloads,
while still exercising the exact code path a real OCR result flows through.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.field_classifier import classify_fields, missing_field_names
from app.schemas import BoundingBox, TextBlock


def block(text: str, confidence: float = 0.9) -> TextBlock:
    return TextBlock(text=text, confidence=confidence,
                      bbox=BoundingBox(x_min=0, y_min=0, x_max=10, y_max=10))


def test_all_fields_present_when_all_patterns_match():
    blocks = [
        block("MRP Rs. 45.00 (Incl. of all taxes)"),
        block("Net Wt. 200g"),
        block("Mfg. Date: 03/2026"),
        block("Consumer Care: 1800-123-4567"),
        block("Marketed by: Acme Foods Pvt Ltd, PIN: 400001"),
    ]
    declarations = classify_fields(blocks)
    assert missing_field_names(declarations) == []
    assert all(d.present for d in declarations)


def test_missing_field_is_reported_not_guessed():
    # No MRP anywhere in these blocks.
    blocks = [
        block("Net Weight 500 ml"),
        block("Packed on 01/2026"),
    ]
    declarations = classify_fields(blocks)
    missing = missing_field_names(declarations)
    assert "mrp" in missing
    mrp_decl = next(d for d in declarations if d.field_name == "mrp")
    assert mrp_decl.present is False
    assert mrp_decl.matched_text is None
    assert "manually" in mrp_decl.notes.lower()


def test_low_confidence_ocr_is_not_trusted():
    # Text technically matches, but confidence is below the trust threshold --
    # this should NOT count as a positive match (avoids false positives from
    # garbled OCR reads).
    blocks = [block("MRP Rs. 99", confidence=0.2)]
    declarations = classify_fields(blocks)
    mrp_decl = next(d for d in declarations if d.field_name == "mrp")
    assert mrp_decl.present is False


def test_devanagari_net_quantity_is_recognised():
    blocks = [block("निवल मात्रा 250 ग्राम")]
    declarations = classify_fields(blocks)
    net_qty = next(d for d in declarations if d.field_name == "net_quantity")
    assert net_qty.present is True


def test_currency_symbol_alone_satisfies_mrp():
    blocks = [block("₹120.50")]
    declarations = classify_fields(blocks)
    mrp_decl = next(d for d in declarations if d.field_name == "mrp")
    assert mrp_decl.present is True


def test_phone_number_satisfies_consumer_care():
    blocks = [block("Call us: 9876543210")]
    declarations = classify_fields(blocks)
    care = next(d for d in declarations if d.field_name == "consumer_care")
    assert care.present is True


def test_empty_input_reports_all_fields_missing():
    declarations = classify_fields([])
    assert len(missing_field_names(declarations)) == 5
