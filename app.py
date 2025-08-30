import cv2
import numpy as np
import aiohttp
import asyncio
import os
import uuid
from fastapi import FastAPI, Query, Request
from fastapi.staticfiles import StaticFiles   # ✅ ye missing tha pehle
import uvicorn

# FastAPI app
app = FastAPI()

# Static serve for outputs folder
OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTDIR), name="outputs")

# Haar cascades
face_cascade = cv2.CascadeClassifier(os.path.join("cascades", "haarcascade_frontalface_default.xml"))
mouth_cascade = cv2.CascadeClassifier(os.path.join("cascades", "haarcascade_mcs_mouth.xml"))

# ---- Helper: download image ----
async def fetch_image(url: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status != 200:
                return None
            data = await resp.read()
            nparr = np.frombuffer(data, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

# ---- Detect mouth ----
def detect_mouth(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]

        mouths = mouth_cascade.detectMultiScale(
            roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        if len(mouths) == 0:
            continue

        # sabse niche wala mouth (bottom-most)
        (mx, my, mw, mh) = max(mouths, key=lambda m: m[1])
        return (x+mx, y+my, mw, mh)
    return None

# ---- Animate mouth ----
def animate_mouth(image, bbox, out_path, frames=24, fps=12):
    (x, y, w, h) = bbox
    roi = image[y:y+h, x:x+w].copy()

    height, width = roi.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (image.shape[1], image.shape[0]))

    for f in range(frames):
        t = f / frames
        open_amt = np.sin(t * np.pi * 2) * 0.5 + 0.5   # oscillates 0-1
        scale = 1.0 + 0.5 * open_amt
        new_h = max(1, int(height * scale))
        resized = cv2.resize(roi, (width, new_h))

        canvas = image.copy()
        center_y = y + height // 2
        new_y1 = int(center_y - new_h // 2)
        new_y1 = max(0, min(new_y1, image.shape[0]-new_h))
        canvas[new_y1:new_y1+new_h, x:x+width] = resized[:image.shape[0]-new_y1, :]

        writer.write(canvas)

    writer.release()

# ---- API endpoint ----
@app.get("/")
def home():
    return {"message": "Mouth animation API running"}

@app.get("/process")
async def process(request: Request, image_url: str = Query(..., description="Public image URL")):
    img = await fetch_image(image_url)
    if img is None:
        return {"error": "Image download failed"}

    bbox = detect_mouth(img)
    if bbox is None:
        return {"error": "No mouth detected!"}

    out_path = os.path.join(OUTDIR, f"anim_{uuid.uuid4().hex}.mp4")
    animate_mouth(img, bbox, out_path)

    # ✅ Full public URL generate karo
    base_url = str(request.base_url).rstrip("/")
    file_name = os.path.basename(out_path)
    return {"video_url": f"{base_url}/outputs/{file_name}"}

# ---- Local run ----
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
