import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from app.comparator import preprocess, compute_similarity
from app.visualizer import save_all_visualizations

app = FastAPI(
    title="Signature Comparator API",
    description="Compare a PAN card signature with a reference sample.",
    version="2.0.0",
)

BASE_OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/compare")
async def compare(
    pan_signature:      UploadFile = File(...),
    sample_signature:   UploadFile = File(...),
    save_visuals:       bool  = True,
    match_threshold:    float = 0.50,   # ← pass in request to override
    mismatch_threshold: float = 0.45,   # ← pass in request to override
):
    allowed = {"image/png", "image/jpeg", "image/jpg"}
    if pan_signature.content_type not in allowed or sample_signature.content_type not in allowed:
        raise HTTPException(status_code=400, detail="Only PNG/JPG images accepted.")

    try:
        pan_bytes    = await pan_signature.read()
        sample_bytes = await sample_signature.read()

        b1 = preprocess(pan_bytes)
        b2 = preprocess(sample_bytes)

        scores = compute_similarity(
        b1, b2,
        match_threshold=match_threshold,
        mismatch_threshold=mismatch_threshold,
    )

        # ── Build subfolder name from filenames ──────────────────────────────
        pan_stem    = os.path.splitext(pan_signature.filename)[0]
        sample_stem = os.path.splitext(sample_signature.filename)[0]
        run_folder  = f"{pan_stem}_vs_{sample_stem}"
        output_dir  = os.path.join(BASE_OUTPUT_DIR, run_folder)

        saved_files = []
        if save_visuals:
            saved_files = save_all_visualizations(
                b1, b2,
                pan_bytes, sample_bytes,
                scores,
                output_dir=output_dir      # ← subfolder passed here
            )

        return JSONResponse(content={
            **scores,
            "output_folder": output_dir,
            "visualizations_saved": saved_files if save_visuals else [],
        })

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")