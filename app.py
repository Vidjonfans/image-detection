import cv2
import numpy as np
import aiohttp
import os
import uuid
import subprocess
from fastapi import FastAPI, Query, Request
from fastapi.staticfiles import StaticFiles
import uvicorn

from animations.mouth_animation import detect_mouth, animate_mouth
from animations.eyes_animation import detect_eyes, animate_eyes

# FastAPI app
app = FastAPI()

OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTDIR), name="outputs")

# ---- Helper: download image ----
async def fetch_image(url: str):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return None
                data = await resp.read()
                nparr = np.frombuffer(data, np.uint8)
                return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[ERROR] fetch_image failed: {e}")
        return None

# ---- Browser-friendly fix ----
def fix_mp4(out_path):
    fixed_path = out_path.replace(".mp4", "_fixed.mp4")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", out_path,
             "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-shortest", fixed_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        os.replace(fixed_path, out_path)
    except Exception as e:
        print("[ERROR] ffmpeg failed:", e)

@app.get("/")
def home():
    return {"message": "Animation API running"}

@app.get("/process")
async def process(
    request: Request,
    image_url: str = Query(..., description="Public image URL"),
    animation: str = Query("mouth", description="Animation type: mouth | eyes")
):
    img = await fetch_image(image_url)
    if img is None:
        return {"error": "Image download failed or invalid URL"}

    out_path = os.path.join(OUTDIR, f"anim_{uuid.uuid4().hex}.mp4")

    if animation == "mouth":
        bbox = detect_mouth(img)
        if bbox is None:
            return {"error": "No mouth detected"}
        frames = animate_mouth(img, bbox, out_path)

    elif animation == "eyes":
        bboxes = detect_eyes(img)
        if len(bboxes) == 0:
            return {"error": "No eyes detected"}
        frames = animate_eyes(img, bboxes, out_path)

    else:
        return {"error": "Unknown animation type"}

    fix_mp4(out_path)

    base_url = str(request.base_url).rstrip("/")
    file_name = os.path.basename(out_path)
    return {
        "video_url": f"{base_url}/outputs/{file_name}",
        "frames_written": frames
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=10000, reload=False)
