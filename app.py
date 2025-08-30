from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os
import uuid
import aiohttp
import cv2
import numpy as np
from typing import Dict

# output folder
OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)

app = FastAPI(title="Mouth Animation API (OpenCV only)")
app.mount("/outputs", StaticFiles(directory=OUTDIR), name="outputs")

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_mcs_mouth.xml")

# ---- Helper to download image ----
async def fetch_image_bytes(url: str) -> bytes:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status != 200:
                raise HTTPException(status_code=400, detail="Failed to download image")
            return await resp.read()

# ---- Detect mouth ----
def detect_mouth(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        mouths = mouth_cascade.detectMultiScale(roi_gray, 1.5, 11)
        if len(mouths) == 0:
            continue
        # pick lowest rectangle (mouth is usually lowest feature in face)
        (mx, my, mw, mh) = max(mouths, key=lambda m: m[1])
        return (x+mx, y+my, mw, mh)
    return None

# ---- Basic mouth animation ----
def animate_mouth(image: np.ndarray, bbox: tuple, out_path: str, frames=24, fps=12):
    (x, y, w, h) = bbox
    roi = image[y:y+h, x:x+w].copy()
    if roi.size == 0:
        raise ValueError("Mouth ROI empty")

    height, width = roi.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (image.shape[1], image.shape[0]))

    for f in range(frames):
        t = f / frames
        # oscillate open/close
        open_amt = np.sin(t * np.pi * 2) * 0.5 + 0.5
        scale = 1.0 + 0.5 * open_amt
        new_h = max(1, int(height * scale))
        resized = cv2.resize(roi, (width, new_h))

        canvas = image.copy()
        center_y = y + height // 2
        new_y1 = int(center_y - new_h // 2)
        new_y1 = max(0, min(new_y1, image.shape[0] - new_h))
        new_y2 = new_y1 + new_h

        try:
            canvas[new_y1:new_y2, x:x+width] = resized
        except Exception:
            h2 = min(new_h, image.shape[0]-new_y1)
            canvas[new_y1:new_y1+h2, x:x+width] = resized[:h2, :]

        writer.write(canvas)

    writer.release()

# ---- API Endpoint ----
@app.post("/generate")
async def generate(payload: Dict):
    url = payload.get("image_url")
    if not url:
        raise HTTPException(status_code=400, detail="image_url required")

    data = await fetch_image_bytes(url)
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    bbox = detect_mouth(img)
    if bbox is None:
        raise HTTPException(status_code=400, detail="No mouth detected")

    filename = f"anim_{uuid.uuid4().hex}.mp4"
    out_path = os.path.join(OUTDIR, filename)

    try:
        animate_mouth(img, bbox, out_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to animate: {e}")

    return {"video_url": f"/outputs/{filename}"}

@app.get("/")
def root():
    return {"msg": "Mouth animation API. POST /generate with {image_url}"}
