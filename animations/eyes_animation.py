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
        t = (np.sin(f / frames * np.pi * 2) + 1) / 2  # oscillates 0 → 1 → 0
        canvas = image.copy()

        for (x, y, w, h) in bboxes:
            roi = image[y:y+h, x:x+w].copy()
            roi_h, roi_w = roi.shape[:2]

            # Blink effect: compress vertically
            new_h = max(1, int(roi_h * (0.3 + 0.7 * t)))  # 0.3 = min eye height
            resized = cv2.resize(roi, (roi_w, new_h))

            # Put resized eye back in the canvas (centered vertically)
            top = y + (roi_h - new_h) // 2
            canvas[y:y+roi_h, x:x+w] = (0,0,0)  # clear area
            canvas[top:top+new_h, x:x+w] = resized

        writer.write(canvas)

    writer.release()
    return frames
