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
        t = f / frames
        blink = (np.sin(t * np.pi * 2) > 0)  # eyes closed half the time

        canvas = image.copy()
        for (x, y, w, h) in bboxes:
            if blink:
                cv2.rectangle(canvas, (x, y+h//3), (x+w, y+2*h//3), (0,0,0), -1)
        writer.write(canvas)

    writer.release()
    return frames
