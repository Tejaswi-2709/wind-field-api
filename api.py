from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List
import os

from main import run_job

app = FastAPI(
    title="Wind Extraction API",
    version="1.0.0"
)

# =====================================================
# PATHS
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# =====================================================
# REQUEST MODEL
# =====================================================

class WindRequest(BaseModel):
    date: str
    bbox: List[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="[min_lon, min_lat, max_lon, max_lat]"
    )
    save_debug: bool = False

# =====================================================
# ROOT ENDPOINT
# =====================================================

@app.get("/")
def root():
    return {
        "message": "Wind Extraction API is running",
        "docs": "/docs"
    }

# =====================================================
# HEALTH CHECK
# =====================================================

@app.get("/health")
def health():
    return {
        "status": "ok"
    }

# =====================================================
# RUN PIPELINE
# =====================================================

@app.post("/extract")
def extract_wind(request: WindRequest):

    result = run_job(
        date=request.date,
        bbox=request.bbox,
        save_debug=request.save_debug
    )

    if result.get("status") != "success":
        raise HTTPException(
            status_code=500,
            detail=result.get("error", "Pipeline failed")
        )

    return {
        "message": "Wind extraction completed",
        "result": {
            "status": result.get("status"),
            "summary": result.get("summary"),

            "plot_url": (
                f"/download/{os.path.basename(result['plot_path'])}"
                if result.get("plot_path")
                else None
            ),

            "csv_url": (
                f"/download/{os.path.basename(result['csv_path'])}"
                if result.get("csv_path")
                else None
            )
        }
    }

# =====================================================
# DOWNLOAD FILE
# =====================================================

@app.get("/download/{filename}")
def download_file(filename: str):

    file_path = os.path.join(OUTPUT_DIR, filename)

    if not os.path.isfile(file_path):
        raise HTTPException(
            status_code=404,
            detail="File not found"
        )

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/octet-stream"
    )