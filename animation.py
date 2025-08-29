import mediapipe as mp
import numpy as np
from moviepy.editor import ImageSequenceClip
import requests
from PIL import Image, ImageDraw
from io import BytesIO
import tempfile
import os

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

def add_animation_on_mouth(image_url):
    # 🔹 Load image from URL
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    w, h = img.size
    img_np = np.array(img)

    # Mediapipe expects RGB np.array
    results = face_mesh.process(img_np)

    if not results.multi_face_landmarks:
        raise Exception("No face detected!")

    landmarks = results.multi_face_landmarks[0]
    MOUTH_IDX = [61, 291, 0, 17, 78, 308]

    mouth_points = []
    for idx in MOUTH_IDX:
        x = int(landmarks.landmark[idx].x * w)
        y = int(landmarks.landmark[idx].y * h)
        mouth_points.append((x, y))

    # 🔹 Animation frames
    frames = []
    for i in range(15):
        frame = img.copy()
        draw = ImageDraw.Draw(frame)

        # Simple animation: pulsating circle at mouth corner
        radius = 5 + i*2
        cx, cy = mouth_points[0]
        draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), outline="red", width=3)

        frames.append(np.array(frame))

    # 🔹 Convert frames to video
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, "mouth_animation.mp4")
    clip = ImageSequenceClip(frames, fps=5)
    clip.write_videofile(video_path, codec="libx264")

    return video_path
