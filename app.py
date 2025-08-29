from flask import Flask, request, jsonify, send_file
import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import time
import os

app = Flask(__name__)

def create_mouth_animation(img_url, output_path="mouth_animation.mp4"):
    # Load Image from URL
    resp = urllib.request.urlopen(img_url)
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
            for idx in [61, 291, 0, 17, 78, 308]:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                mouth_points.append((x, y))

    # Animation create
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 15, (w, h))

    for frame_id in range(30):  # ~2 sec animation
        frame = image.copy()
        if mouth_points:
            pts = np.array(mouth_points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            alpha = 0.3 + 0.2*np.sin(frame_id/5.0)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], (0, 0, 255))  # red glow
            cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
        out.write(frame)

    out.release()
    return output_path

@app.route("/", methods=["GET"])
def home():
    return {"message": "Mouth Animation API is running!"}

@app.route("/animate", methods=["POST"])
def animate():
    data = request.json
    img_url = data.get("image_url")
    if not img_url:
        return jsonify({"error": "image_url is required"}), 400

    output_file = f"output_{int(time.time())}.mp4"
    path = create_mouth_animation(img_url, output_file)
    return send_file(path, mimetype="video/mp4", as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
