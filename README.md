# ComplyScan — Phase 1 MVP

AI-based Legal Metrology label compliance scanner. This is **Phase 1** only:
OCR + mandatory-field extraction. It answers *"what declarations did we find
on this label, and what's missing"* — it does not yet apply category-specific
rules/exemptions (Phase 2), measure font size in mm (Phase 3), or detect MRP
sticker tampering (Phase 3/4). Those are separate, later modules that build
on this extraction output.

## What's actually been tested vs. what hasn't

Everything in this repo was written and tested **except one thing**: real
OCR inference. Here's the exact status, honestly:

| Component | Status |
|---|---|
| Image preprocessing (deskew, denoise, glare correction) | ✅ Tested on real generated images — deskew verified to correct a 4° rotation |
| Field classifier (regex/keyword matching → 5 mandatory fields) | ✅ 7 unit tests passing |
| FastAPI `/scan` endpoint (upload validation, error handling, response shape) | ✅ 6 integration tests passing (OCR mocked) |
| PaddleOCR model inference itself | ⚠️ **Not run end-to-end here.** PaddleOCR downloads its detection/recognition model weights from `paddleocr.bj.bcebos.com` on first use — the sandbox this was built in has a locked-down network allowlist that doesn't include that host (confirmed: got an HTTP 403 from the download step). This will work normally on your own machine with regular internet access. |

In other words: the *system* — preprocessing, the classification logic, the
API contract, error handling — is real and verified. The one link I
couldn't personally pull the trigger on is the actual OCR model call, because
of where I was running it, not because of a bug. First run on your machine
will download ~15-40MB of model weights automatically; after that it's cached
in `~/.paddleocr/`.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run it

```bash
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000/docs for the interactive Swagger UI, or:

```bash
curl -X POST http://localhost:8000/scan \
  -F "file=@sample_data/sample_label_straight.jpg"
```

First request will be slow (model download + load). Subsequent requests
reuse the cached engine (see `app/ocr_engine.py` — `lru_cache`).

## Run the tests

```bash
pip install pytest httpx
python3 -m pytest tests/ -v
```

13 tests, all passing, no network required (OCR is mocked in `test_api.py`
deliberately — see the docstring there for why).

## Testing with real images

The generated sample labels in `sample_data/` are synthetic (clean font,
white background) — good for testing the *pipeline*, not OCR accuracy on
real photos. To actually validate OCR quality:

1. Photograph a few real packaged products (or find product images online).
2. `curl -X POST http://localhost:8000/scan -F "file=@your_photo.jpg"`
3. Compare `raw_text_blocks` in the response against what's actually on the
   label. Low-confidence or missing blocks tell you where preprocessing
   needs work (e.g. add perspective correction for curved bottles — see the
   roadmap note at the bottom of `app/preprocessing.py`).

## Project structure

```
complyscan/
├── app/
│   ├── main.py              FastAPI app, /scan endpoint
│   ├── preprocessing.py     OpenCV: resize, denoise, glare/contrast, deskew
│   ├── ocr_engine.py        PaddleOCR wrapper (cached engine, TextBlock output)
│   ├── field_classifier.py  Regex/keyword mapping: text blocks → 5 mandatory fields
│   └── schemas.py           Pydantic response models
├── tests/
│   ├── test_field_classifier.py   Pure logic tests, no OCR
│   ├── test_api.py                API tests, OCR mocked
│   └── make_sample_label.py       Generates synthetic test images
├── sample_data/              Generated test images (gitignored contents are fine to keep for demo)
└── requirements.txt
```

## What Phase 1 deliberately does NOT do

- **Category/exemption logic** — a biscuit and an imported industrial
  chemical have different mandatory fields. Phase 1 checks the same 5 core
  fields on everything. Phase 2's rule engine (JSON/YAML-config, not
  hardcoded) is where category-specific rules and exemptions belong.
- **Font-size (mm) measurement** — needs a physical scale reference in the
  photo (Phase 3).
- **MRP tamper/sticker detection** — needs OpenCV texture/edge forensics on
  top of OCR, not OCR alone (Phase 3/4).
- **Perspective correction for curved surfaces** (bottles/jars) — flagged as
  a known gap with a specific implementation note in `preprocessing.py`.
  Left out because unreliable corner-detection needs its own validation
  step, not a silent bolt-on.

## Before you deploy this anywhere real

`app/main.py` has `allow_origins=["*"]` in CORS middleware — that's fine for
local dev, not for anything public. Also: there's no auth on `/scan` yet
(Phase 5 in the roadmap adds role-based access).
