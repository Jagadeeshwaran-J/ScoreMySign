# ✍️ Signature Comparator API

A **FastAPI-based REST service** that compares two signature images (e.g., a PAN card signature vs. a reference sample) using a multi-method similarity pipeline. It returns a composite similarity score, a human-readable verdict, and optional visual analysis charts.

---

## 🧠 How It Works

Each pair of signatures is processed through a  **5-method similarity pipeline** , and the results are combined into a single weighted composite score:

| Method                | Weight | Description                                                     |
| --------------------- | ------ | --------------------------------------------------------------- |
| **SSIM**        | 20%    | Structural Similarity Index — pixel-level structure comparison |
| **pHash**       | 20%    | Perceptual Hash — DCT-based frequency fingerprint              |
| **Contour**     | 15%    | Number of contours (ink strokes) comparison                     |
| **Ink Density** | 15%    | Ratio of ink pixels between both signatures                     |
| **Hausdorff**   | 30%    | Skeleton-based shape distance — stroke shape analysis          |

---

## 🧾 Verdicts

| Score Range  | Verdict                 |
| ------------ | ----------------------- |
| ≥ 0.50      | `MATCH`               |
| 0.45 – 0.50 | `LIKELY_MISMATCH`     |
| < 0.45       | `FORGERY_OR_MISMATCH` |

---

## 📁 Project Structure

```
├── app/
│   ├── __init__.py
│   ├── comparator.py        # Similarity logic (SSIM, pHash, contour, ink, Hausdorff)
│   ├── main.py              # FastAPI app and /compare endpoint
│   └── visualizer.py        # Visualization charts generator
├── images/                  # Mount your input images here
├── outputs/                 # Generated visual analysis PNGs land here
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 🚀 Running the Project

### Option 1 — Docker (Recommended)

**Prerequisites:** Docker Desktop installed and running.

```bash
# 1. Clone the repository
git clone https://github.com/Jagadeeshwaran-J/ScoreMySign.git
cd ScoreMySign

# 2. Build and start the container
docker compose up --build

# 3. The API is now live at:
# http://localhost:8000/
```

> Output visualizations will be saved to the `./outputs/` folder on your machine.

---

### Option 2 — Local Python (No Docker)

**Prerequisites:** Python 3.11+

```bash
# 1. Clone the repository
git clone https://github.com/Jagadeeshwaran-J/ScoreMySign.git
cd your-ScoreMySign

# 2. Create and activate a virtual environment
python -m venv venv
or
uv venv --python 312

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# 3. Install dependencies
uv pip install -r requirements.txt

# 4. Run the server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 📡 API Endpoints

### `GET /health`

Returns service health status.

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

---

### `POST /compare`

Compares two signature images and returns similarity scores.

#### Form Parameters

| Parameter              | Type  | Required | Default  | Description                               |
| ---------------------- | ----- | -------- | -------- | ----------------------------------------- |
| `pan_signature`      | file  | ✅       | —       | PAN card signature image (PNG/JPG)        |
| `sample_signature`   | file  | ✅       | —       | Reference signature image (PNG/JPG)       |
| `save_visuals`       | bool  | ❌       | `true` | Save visualization charts to `outputs/` |
| `match_threshold`    | float | ❌       | `0.50` | Score above which = MATCH                 |
| `mismatch_threshold` | float | ❌       | `0.45` | Score below which = FORGERY_OR_MISMATCH   |

---

### Example Request

```bash
curl -X POST http://localhost:8000/compare \
  -F "pan_signature=@/path/to/pan_sig.png" \
  -F "sample_signature=@/path/to/sample_sig.png"
```

---

### Example Response

```json
{
  "ssim": 0.8123,
  "phash": 0.7656,
  "contour": 0.9000,
  "ink": 0.8741,
  "hausdorff": 0.7812,
  "composite": 0.8134,
  "verdict": "MATCH",
  "thresholds": { "match": 0.55, "mismatch": 0.50 },
  "output_folder": "outputs/pan_sig_vs_sample_sig",
  "visualizations_saved": [
    "outputs/pan_sig_vs_sample_sig/01_preprocessing_pipeline.png",
    "outputs/pan_sig_vs_sample_sig/02_method_ssim.png",
    "outputs/pan_sig_vs_sample_sig/03_method_phash.png",
    "outputs/pan_sig_vs_sample_sig/04_method_contour.png",
    "outputs/pan_sig_vs_sample_sig/05_method_ink_density.png",
    "outputs/pan_sig_vs_sample_sig/06_method_hausdorff.png",
    "outputs/pan_sig_vs_sample_sig/07_final_dashboard.png"
  ]
}
```

---

## 📊 Visualizations

When `save_visuals=true`, the API generates 7 detailed PNG charts per comparison run:

| File                              | Content                          |
| --------------------------------- | -------------------------------- |
| `01_preprocessing_pipeline.png` | Step-by-step image preprocessing |
| `02_method_ssim.png`            | SSIM difference heatmap          |
| `03_method_phash.png`           | DCT hash comparison              |
| `04_method_contour.png`         | Contour detection overlay        |
| `05_method_ink_density.png`     | Ink density comparison           |
| `06_method_hausdorff.png`       | Skeleton stroke distance         |
| `07_final_dashboard.png`        | Final summary dashboard          |

---

## 📦 Dependencies

Key libraries used:

* fastapi + uvicorn — API framework
* opencv-python-headless — Image processing
* scikit-image — SSIM & skeletonization
* scipy — Hausdorff distance
* numpy, Pillow — Image utilities
* matplotlib — Visualization charts

---

## ⚙️ Configuration

Set a custom output directory using the `OUTPUT_DIR` environment variable:

```bash
OUTPUT_DIR=/custom/path uvicorn app.main:app --port 8000
```


