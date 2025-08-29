from flask import Flask, request, jsonify, send_from_directory
import cv2, numpy as np, math, requests, os, uuid

app = Flask(__name__)

# folder jaha videos save honge
OUTPUT_DIR = "static"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def animate_face(image_url, output_path):
    # image download
    resp = requests.get(image_url, stream=True)
    arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        raise Exception("Image load failed")

    h, w = img.shape[:2]

    # Haarcascade face detection
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40,40))

    if len(faces) == 0:
        # fallback agar face na mile
        fw = int(w * 0.4); fh = int(h * 0.5)
        fx, fy = w//2-fw//2, h//2-fh//2
        faces = [(fx, fy, fw, fh)]
    (x, y, fw, fh) = faces[0]

    face_crop = img[y:y+fh, x:x+fw].copy()
    fc_h, fc_w = face_crop.shape[:2]

    # alpha mask
    mask = np.zeros((fc_h, fc_w), dtype=np.uint8)
    cv2.ellipse(mask, (fc_w//2, fc_h//2),
                (int(fc_w*0.48), int(fc_h*0.48)), 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (11,11), 0)
    alpha = mask.astype(np.float32)/255.0
    alpha = cv2.merge([alpha, alpha, alpha])

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

@app.route("/")
def home():
    return "✅ Face Animation API is running!"

@app.route("/animate")
def animate():
    image_url = request.args.get("url")
    if not image_url:
        return jsonify({"error": "url parameter required"}), 400

    # unique filename
    filename = f"{uuid.uuid4().hex}.mp4"
    output_path = os.path.join(OUTPUT_DIR, filename)

    try:
        animate_face(image_url, output_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    video_url = request.host_url + f"static/{filename}"
    return jsonify({"video_url": video_url})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
