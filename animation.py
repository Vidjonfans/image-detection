import cv2
import numpy as np
from moviepy.editor import ImageSequenceClip
import requests
from PIL import Image
from io import BytesIO
import tempfile
import os

# Haarcascade mouth detector (pre-trained xml file)
MOUTH_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

def add_animation_on_mouth(image_url):
    # Load image from URL
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    frame = np.array(img)

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    mouths = MOUTH_CASCADE.detectMultiScale(gray, 1.8, 20)

    if len(mouths) == 0:
        raise Exception("No mouth detected!")

    (x, y, w, h) = mouths[0]

    # Animation frames
    frames = []
    for i in range(15):
        temp = frame.copy()
        radius = 5 + i*2
        cx, cy = x + w//2, y + h//2
        cv2.circle(temp, (cx, cy), radius, (255, 0, 0), 3)
        frames.append(temp)

    # Save video
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, "mouth_animation.mp4")
    clip = ImageSequenceClip([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames], fps=5)
    clip.write_videofile(video_path, codec="libx264")

    return video_path
