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
def detect_eyes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None

    eyes_bboxes = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(
            roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
        )
        for (ex, ey, ew, eh) in eyes:
            eyes_bboxes.append((x+ex, y+ey, ew, eh))
    return eyes_bboxes if eyes_bboxes else None


# ---- Animate mouth ----
def animate_eyes(image, bboxes, out_path, frames=24, fps=12):
    canvases = []
    height, width = image.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    for f in range(frames):
        t = f / frames
        blink_amt = (np.sin(t * np.pi * 4) + 1) / 2  # oscillates 0-1 (blink)
        canvas = image.copy()

        for (x, y, w, h) in bboxes:
            eye_roi = image[y:y+h, x:x+w].copy()
            new_h = max(1, int(h * (1 - 0.8 * blink_amt)))  # blink squish
            resized_eye = cv2.resize(eye_roi, (w, new_h))
            center_y = y + h // 2
            new_y1 = int(center_y - new_h // 2)
            canvas[new_y1:new_y1+new_h, x:x+w] = resized_eye[:image.shape[0]-new_y1, :]
        
        writer.write(canvas)
    
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
@app.get("/process_eyes")
async def process_eyes(request: Request, image_url: str):
    img = await fetch_image(image_url)
    if img is None:
        return {"error": "Image download failed or invalid URL"}

    bboxes = detect_eyes(img)
    if bboxes is None:
        return {"error": "No eyes detected!"}

    out_path = os.path.join(OUTDIR, f"blink_{uuid.uuid4().hex}.mp4")
    animate_eyes(img, bboxes, out_path)
    fix_mp4(out_path)

    base_url = str(request.base_url).rstrip("/")
    file_name = os.path.basename(out_path)
    return {
        "video_url": f"{base_url}/outputs/{file_name}",
        "eyes_detected": len(bboxes)
    }



    

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

