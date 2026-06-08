from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import os

from main import run_job

app = FastAPI(title="Wind Extraction API")

OUTPUT_DIR = "outputs"


# =========================
# REQUEST MODEL
# =========================
class WindRequest(BaseModel):
    date: str
    bbox: List[float]
    save_debug: bool = False


# =========================
# RUN PIPELINE
# =========================
@app.post("/extract")
def extract_wind(request: WindRequest):

    result = run_job(
        date=request.date,
        bbox=request.bbox,
        save_debug=request.save_debug
    )

    if result["status"] != "success":
        return {
            "message": "Pipeline failed",
            "error": result.get("error")
        }

    return {
        "message": "Wind extraction completed",
        "result": {
            "status": result["status"],
            "summary": result["summary"],

            # download links (IMPORTANT CHANGE)
            "plot_url": f"/download/{os.path.basename(result['plot_path'])}" if result.get("plot_path") else None,
            "csv_url": f"/download/{os.path.basename(result['csv_path'])}"
        }
    }


# =========================
# DOWNLOAD ENDPOINT
# =========================
@app.get("/download/{filename}")
def download_file(filename: str):

    file_path = os.path.join(OUTPUT_DIR, filename)

    if not os.path.exists(file_path):
        return {"error": "File not found"}

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/octet-stream"
    )