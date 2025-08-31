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

    h_img, w_img = image.shape[:2]

    for f in range(frames):
        t = (np.sin(f / frames * np.pi))  # 0 → 1 → 0 smooth blink

        canvas = image.copy()

        for (x, y, w, h) in bboxes:
            eye_center = y + h // 2

            # Blink amount
            close_amt = int(h * t * 0.5)  # 50% closure

            # Upper eyelid line
            top_start = y
            top_end = y + close_amt

            # Lower eyelid line
            bottom_start = y + h
            bottom_end = y + h - close_amt

            # Draw filled polygons (eyelids) with skin-like color
            avg_color = np.mean(image[y:y+h, x:x+w].reshape(-1,3), axis=0)
            color = tuple(map(int, avg_color))

            cv2.rectangle(canvas, (x, top_start), (x+w, top_end), color, -1)
            cv2.rectangle(canvas, (x, bottom_end), (x+w, bottom_start), color, -1)

        writer.write(canvas)

    writer.release()
    return frames
