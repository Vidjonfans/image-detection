import cv2
import numpy as np
import math
from pathlib import Path

def animate_face(input_path, output_path="face_animated.mp4"):
    img = cv2.imread(str(input_path))
    if img is None:
        raise FileNotFoundError("Input image not found.")

    # Resize for speed
    max_w = 640
    h, w = img.shape[:2]
    if w > max_w:
        img = cv2.resize(img, (max_w, int(h*max_w/w)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    # Haarcascade face detection
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40,40))

    if len(faces) == 0:
        print("⚠️ No face found — using center crop fallback.")
        fw = int(w * 0.4); fh = int(h * 0.5)
        fx, fy = w//2-fw//2, h//2-fh//2
        faces = [(fx, fy, fw, fh)]
    (x, y, fw, fh) = faces[0]

    # Crop face
    face_crop = img[y:y+fh, x:x+fw].copy()
    fc_h, fc_w = face_crop.shape[:2]

    # Alpha mask
    mask = np.zeros((fc_h, fc_w), dtype=np.uint8)
    cv2.ellipse(mask, (fc_w//2, fc_h//2), (int(fc_w*0.48), int(fc_h*0.48)), 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (11,11), 0)
    alpha = mask.astype(np.float32)/255.0
    alpha = cv2.merge([alpha, alpha, alpha])

    # Video writer
    fps, frames = 24, 30
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w,h))

    center = (fc_w/2, fc_h/2)
    for i in range(frames):
        t = i/frames
        dy = int(math.sin(2*math.pi*t*2)*fh*0.06)
        dx = int(math.sin(2*math.pi*t*1.5)*fw*0.02)
        angle = math.sin(2*math.pi*t*1.2)*6
        scale = 1.0 + math.sin(2*math.pi*t*1.2)*0.02

        M = cv2.getRotationMatrix2D(center, angle, scale)
        transformed = cv2.warpAffine(face_crop, M, (fc_w, fc_h),
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        frame = img.copy()
        tx = max(0, min(w-fc_w, x+dx))
        ty = max(0, min(h-fc_h, y+dy))

        roi = frame[ty:ty+fc_h, tx:tx+fc_w].astype(np.float32)
        blended = (transformed.astype(np.float32)*alpha + roi*(1.0-alpha)).astype(np.uint8)
        frame[ty:ty+fc_h, tx:tx+fc_w] = blended
        out.write(frame)

    out.release()
    print(f"✅ Animation saved to {output_path}")

# Example run
if __name__ == "__main__":
    animate_face("sample.png", "face_animated.mp4")
