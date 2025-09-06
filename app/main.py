from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil, uuid
from .pipeline import process_no_crop

app = FastAPI(title="FOMM Mask Animation API")

BASE = Path(__file__).parent
STATIC = BASE / "static"
OUT = STATIC / "outputs"
WEIGHTS = BASE / "weights"

app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

@app.get("/")
def root():
    return {"status": "ok", "endpoints": ["/animate"]}

@app.post("/animate")
async def animate(source_image: UploadFile = File(...),
                  driving_video: UploadFile = File(...),
                  fps: int = 25):
    job_id = uuid.uuid4().hex[:10]
    workdir = OUT / job_id
    workdir.mkdir(parents=True, exist_ok=True)
    src = workdir / "source.jpg"
    drv = workdir / "driving.mp4"
    outp = workdir / "result.mp4"

    with src.open("wb") as f:
        shutil.copyfileobj(source_image.file, f)
    with drv.open("wb") as f:
        shutil.copyfileobj(driving_video.file, f)

    result = process_no_crop(src, drv, WEIGHTS, outp, fps=fps)
    url = f"/static/outputs/{job_id}/result.mp4"
    return {"url": url, "job_id": job_id}
