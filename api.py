from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import random
import time
import math

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

    # simulate processing time
    time.sleep(2)

    min_lon, min_lat, max_lon, max_lat = request.bbox

    # generate sample wind vectors
    sample_vectors = []

    for i in range(5):

        lat = round(random.uniform(min_lat, max_lat), 4)
        lon = round(random.uniform(min_lon, max_lon), 4)

        speed = round(random.uniform(5, 15), 2)
        direction = round(random.uniform(0, 360), 2)

        u = round(speed * math.cos(math.radians(direction)), 2)
        v = round(speed * math.sin(math.radians(direction)), 2)

        sample_vectors.append({
            "lat": lat,
            "lon": lon,
            "wind_speed": speed,
            "wind_direction": direction,
            "u": u,
            "v": v
        })

    return {
        "status": "success",
        "message": "Wind extraction completed",
        "input_date": request.date,
        "bbox": request.bbox,
        "summary": {
            "points_processed": len(sample_vectors),
            "mean_wind_speed": round(
                sum(v["wind_speed"] for v in sample_vectors) / len(sample_vectors),
                2
            )
        },
        "sample_vectors": sample_vectors
    }