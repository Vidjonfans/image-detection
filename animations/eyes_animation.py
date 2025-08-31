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
        # blink factor: 0 -> open, 1 -> closed
        t = (np.sin(f / frames * np.pi))  
        # restrict closure between 25%–55%
        t = 0.25 + t * 0.30  

        canvas = image.copy()

        for (x, y, w, h) in bboxes:
            roi = image[y:y+h, x:x+w].copy()

            # how much to shrink
            close_amt = int(h * t)

            # create ellipse mask (eye-shape)
            mask = np.zeros((h, w), dtype=np.uint8)
            center = (w//2, h//2)
            axes = (w//2, h//2)  # ellipse size
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

            # shrink vertically
            new_h = max(1, h - close_amt)
            resized = cv2.resize(roi, (w, new_h))

            # paste resized eye into blank eye ROI
            blank = roi.copy()
            blank[:] = 0
            top = (h - new_h)//2
            blank[top:top+new_h, :] = resized

            # apply mask (keep only eye-shape area)
            masked_eye = cv2.bitwise_and(blank, blank, mask=mask)

            # merge with canvas
            canvas[y:y+h, x:x+w] = cv2.addWeighted(
                canvas[y:y+h, x:x+w], 1, masked_eye, 1, 0
            )

        writer.write(canvas)

    writer.release()
    return frames
