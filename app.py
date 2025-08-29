from flask import Flask, request, jsonify
import cv2, numpy as np, math, requests, os, uuid, mediapipe as mp

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello 👋 Face Animation API is running on Render ✅"

@app.route('/animate', methods=['POST'])
def animate_face():
    try:
        data = request.get_json()
        img_url = data.get("image_url")

        if not img_url:
            return jsonify({"error": "Please provide image_url"}), 400

        # Download image
        resp = requests.get(img_url)
        img_arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

        # Mediapipe face detection
        mp_face = mp.solutions.face_detection
        detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)

        results = detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if not results.detections:
            return jsonify({"error": "No face detected"}), 404

        # Animate (simple mouth movement demo)
        h, w, _ = img.shape
        out_path = f"static/{uuid.uuid4()}.mp4"
        os.makedirs("static", exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(out_path, fourcc, 10, (w, h))

        for i in range(30):
            frame = img.copy()
            y = 50 * math.sin(i / 5.0)
            cv2.ellipse(frame, (w//2, h//2 + 50), (80, int(20 + y)), 0, 0, 360, (0,0,255), -1)
            video.write(frame)

        video.release()

        return jsonify({
            "status": "success",
            "video_url": f"https://{request.host}/{out_path}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

