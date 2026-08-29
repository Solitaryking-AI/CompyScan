"""
Generates a synthetic product-label image for local testing, since we
don't have a real photographed label handy. Not a substitute for testing
on real photos -- just enough to sanity-check the preprocessing pipeline
end-to-end (decode -> resize -> denoise -> CLAHE -> deskew -> valid image out).
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path(__file__).resolve().parent.parent / "sample_data"
OUT_DIR.mkdir(exist_ok=True)


def make_label(path: Path, rotate_degrees: float = 0.0) -> None:
    img = Image.new("RGB", (900, 600), color="white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
        font_small = font

    lines = [
        ("MRP Rs. 45.00 (Incl. of all taxes)", font),
        ("Net Wt. 200g", font),
        ("Mfg. Date: 03/2026", font_small),
        ("Consumer Care: 1800-123-4567", font_small),
        ("Marketed by: Acme Foods Pvt Ltd, Mumbai, PIN: 400001", font_small),
    ]
    y = 60
    for text, f in lines:
        draw.text((60, y), text, fill="black", font=f)
        y += 80

    if rotate_degrees:
        img = img.rotate(rotate_degrees, expand=True, fillcolor="white")

    img.save(path, "JPEG", quality=90)


if __name__ == "__main__":
    make_label(OUT_DIR / "sample_label_straight.jpg", rotate_degrees=0)
    make_label(OUT_DIR / "sample_label_skewed.jpg", rotate_degrees=4)
    print(f"Sample labels written to {OUT_DIR}")
