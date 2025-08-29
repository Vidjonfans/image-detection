from flask import Flask, request, jsonify, send_file
from animation import add_animation_on_mouth

app = Flask(__name__)

@app.route("/animate", methods=["POST"])
def animate():
    data = request.get_json()
    image_url = data.get("image_url")

    if not image_url:
        return jsonify({"error": "image_url is required"}), 400

    try:
        video_path = add_animation_on_mouth(image_url)
        return send_file(video_path, mimetype="video/mp4")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
