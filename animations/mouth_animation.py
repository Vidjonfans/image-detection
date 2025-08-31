import cv2
import numpy as np
import os
import subprocess
import uuid

CASCADE_DIR = "cascades"
face_cascade_path = os.path.join(CASCADE_DIR, "haarcascade_frontalface_default.xml")
mouth_cascade_path = os.path.join(CASCADE_DIR, "haarcascade_mcs_mouth.xml")

face_cascade = cv2.CascadeClassifier(face_cascade_path)
mouth_cascade = cv2.CascadeClassifier(mouth_cascade_path)

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

def animate_mouth(image, bbox, out_path, frames=24, fps=12):
    (x, y, w, h) = bbox
    roi = image[y:y+h, x:x+w].copy()

    height, width = roi.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (image.shape[1], image.shape[0]))

    written = 0
    for f in range(frames):
        t = f / frames
        open_amt = np.sin(t * np.pi * 2) * 0.5 + 0.5
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

    return written
