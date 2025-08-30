import cv2
import numpy as np
import aiohttp
import asyncio
import os
import uuid

# Create output folder
OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)

# Haar cascades (make sure these XML files exist in cascades/ folder)
face_cascade = cv2.CascadeClassifier(os.path.join("cascades", "haarcascade_frontalface_default.xml"))
mouth_cascade = cv2.CascadeClassifier(os.path.join("cascades", "haarcascade_mcs_mouth.xml"))

# ---- Helper: download image from URL ----
async def fetch_image(url: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
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

        # detect mouths inside face region
        mouths = mouth_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(mouths) == 0:
            continue

        # take lowest mouth rectangle (because mouth usually at bottom of face)
        (mx, my, mw, mh) = max(mouths, key=lambda m: m[1])
        return (x+mx, y+my, mw, mh)
    return None

# ---- Animate mouth open/close ----
def animate_mouth(image, bbox, out_path, frames=24, fps=12):
    (x, y, w, h) = bbox
    roi = image[y:y+h, x:x+w].copy()

    height, width = roi.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (image.shape[1], image.shape[0]))

    for f in range(frames):
        t = f / frames
        open_amt = np.sin(t * np.pi * 2) * 0.5 + 0.5   # oscillates between 0-1
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

# ---- Run demo ----
async def main():
    url = "https://i.ibb.co/G4PMfZrB/bea9fad974589140757bca1df99d1908-removebg-preview.png"
    img = await fetch_image(url)
    bbox = detect_mouth(img)
    if bbox is None:
        print("No mouth detected!")
        return
    out_path = os.path.join(OUTDIR, f"anim_{uuid.uuid4().hex}.mp4")
    animate_mouth(img, bbox, out_path)
    print("Video saved at:", out_path)

if __name__ == "__main__":
    asyncio.run(main())
