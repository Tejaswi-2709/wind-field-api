from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Wind Extraction API")


class WindRequest(BaseModel):
    date: str
    bbox: List[float]
    save_debug: bool = False


@app.get("/")
def root():
    return {
        "message": "Wind Field API Running"
    }


@app.post("/extract")
def extract_wind(request: WindRequest):

    return {
        "message": "Request received successfully",
        "input": {
            "date": request.date,
            "bbox": request.bbox,
            "save_debug": request.save_debug
        },
        "status": "success"
    }