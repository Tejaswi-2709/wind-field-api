from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from main import run_job

app = FastAPI(title="Wind Extraction API")


class WindRequest(BaseModel):
    date: str
    bbox: List[float]
    save_debug: bool = False


@app.post("/extract")
def extract_wind(request: WindRequest):
    result = run_job(
        date=request.date,
        bbox=request.bbox,
        save_debug=request.save_debug
    )

    return {
        "message": "Wind extraction completed",
        "result": result
    }