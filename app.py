import cv2
import mediapipe as mp
import numpy as np
import urllib.request

# Image URL
url = "https://i.ibb.co/v4ykcTp4/e8283822a24d72765a6f51a801e33525-1-1-2.png"
resp = urllib.request.urlopen(url)
image = np.asarray(bytearray(resp.read()), dtype=np.uint8)
image = cv2.imdecode(image, cv2.IMREAD_COLOR)

h, w, _ = image.shape

# Mediapipe Setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# Detect face landmarks
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = face_mesh.process(rgb)

mouth_points = []
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        for idx in [61, 291, 0, 17, 78, 308]:  # कुछ mouth के landmarks
            x = int(face_landmarks.landmark[idx].x * w)
            y = int(face_landmarks.landmark[idx].y * h)
            mouth_points.append((x, y))

# Animation create (mouth area पर glow effect)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("mouth_animation.mp4", fourcc, 15, (w, h))

for frame_id in range(30):  # 2 sec animation
    frame = image.copy()

    if mouth_points:
        pts = np.array(mouth_points, np.int32)
        pts = pts.reshape((-1, 1, 2))

        # thickness/alpha change for animation
        alpha = 0.3 + 0.2*np.sin(frame_id/5.0)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (0, 0, 255))  # red glow
        cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)

    out.write(frame)

out.release()
print("✅ Animation saved as mouth_animation.mp4")
