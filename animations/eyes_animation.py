# eyes_animation.py
import cv2
import numpy as np
import os

CASCADE_DIR = "cascades"
eye_cascade_path = os.path.join(CASCADE_DIR, "haarcascade_eye.xml")

eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

def detect_eyes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    return eyes

def animate_eyes(image, bboxes, out_path, frames=24, fps=12):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (image.shape[1], image.shape[0]))

    for f in range(frames):
        # blink curve (0=open, 1=closed, 0=open)
        t = (np.sin(f / frames * np.pi) ** 2)

        canvas = image.copy()

        for (x, y, w, h) in bboxes:
            roi = image[y:y+h, x:x+w].copy()
            roi_h, roi_w = roi.shape[:2]

            # Calculate how much to close
            close_amt = int(h * t * 0.7)  # max 70% closure

            # Copy background under eye
            bg_patch = image[y:y+h, x:x+w].copy()
            canvas[y:y+h, x:x+w] = bg_patch

            # Shrink ROI vertically
            new_h = max(1, roi_h - close_amt)
            resized = cv2.resize(roi, (roi_w, new_h))

            # Position eye at center vertically
            top = y + (roi_h - new_h) // 2
            canvas[top:top+new_h, x:x+w] = resized

        writer.write(canvas)

    writer.release()
    return frames
