import cv2
import numpy as np
import aiohttp
import os
import uuid
import subprocess
from fastapi import FastAPI, Query, Request
from fastapi.staticfiles import StaticFiles
import uvicorn

# FastAPI app
app = FastAPI()

# Static serve for outputs folder
OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTDIR), name="outputs")

# Haar cascades path check
CASCADE_DIR = "cascades"
face_cascade_path = os.path.join(CASCADE_DIR, "haarcascade_frontalface_default.xml")
mouth_cascade_path = os.path.join(CASCADE_DIR, "haarcascade_mcs_mouth.xml")

if not os.path.exists(face_cascade_path):
    raise FileNotFoundError(f"Face cascade not found: {face_cascade_path}")
if not os.path.exists(mouth_cascade_path):
    raise FileNotFoundError(f"Mouth cascade not found: {mouth_cascade_path}")

# Haar cascades load
face_cascade = cv2.CascadeClassifier(face_cascade_path)
mouth_cascade = cv2.CascadeClassifier(mouth_cascade_path)

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

    written = 0
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
        written += 1

    writer.release()

    # ✅ Duration calculate karo
    cap = cv2.VideoCapture(out_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_val = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    duration = 0
    if fps_val > 0:
        duration = total_frames / fps_val

    return written, duration

# ---- Browser-friendly fix (ffmpeg re-encode) ----
def fix_mp4(out_path):
    fixed_path = out_path.replace(".mp4", "_fixed.mp4")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", out_path,
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-shortest",
                fixed_path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        os.replace(fixed_path, out_path)  # overwrite original
        print("[INFO] MP4 re-encoded for browser compatibility")
    except Exception as e:
        print("[ERROR] ffmpeg failed:", e)

# ---- API endpoint ----
@app.get("/")
def home():
    return {"message": "Mouth animation API running"}

@app.get("/process")
async def process(request: Request, image_url: str = Query(..., description="Public image URL")):
    img = await fetch_image(image_url)
    if img is None:
        return {"error": "Image download failed or invalid URL"}

    bbox = detect_mouth(img)
    if bbox is None:
        return {"error": "No mouth detected!"}

    out_path = os.path.join(OUTDIR, f"anim_{uuid.uuid4().hex}.mp4")
    frame_count, duration = animate_mouth(img, bbox, out_path)

    # ✅ Browser friendly bnao
    fix_mp4(out_path)

    # ✅ Full public URL generate karo
    base_url = str(request.base_url).rstrip("/")
    file_name = os.path.basename(out_path)
    return {
        "video_url": f"{base_url}/outputs/{file_name}",
        "frames_written": frame_count,
        "duration_seconds": duration
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=10000, reload=False)

