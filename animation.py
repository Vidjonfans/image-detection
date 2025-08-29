from flask import Flask, request, send_file, render_template_string, jsonify
import cv2
import numpy as np
import urllib.request
import os
import math
import tempfile
import io
import base64
from datetime import datetime

app = Flask(__name__)

# HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>OpenCV Mouth Animation</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #555;
        }
        input[type="url"] {
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        input[type="url"]:focus {
            outline: none;
            border-color: #667eea;
        }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
            width: 100%;
            transition: transform 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
        }
        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        .result {
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            display: none;
        }
        .error {
            background: #ffebee;
            color: #c62828;
            border: 1px solid #ffcdd2;
        }
        .success {
            background: #e8f5e8;
            color: #2e7d32;
            border: 1px solid #c8e6c9;
        }
        .loading {
            text-align: center;
            color: #666;
        }
        .spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        video {
            width: 100%;
            max-width: 500px;
            border-radius: 8px;
            margin-top: 10px;
        }
        .download-link {
            display: inline-block;
            margin-top: 10px;
            padding: 8px 16px;
            background: #28a745;
            color: white;
            text-decoration: none;
            border-radius: 5px;
        }
        .demo-image {
            width: 100%;
            max-width: 300px;
            border-radius: 8px;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎬 OpenCV Mouth Animation Generator</h1>
        
        <div class="form-group">
            <label for="imageUrl">Image URL (Direct link to image):</label>
            <input type="url" id="imageUrl" placeholder="https://example.com/image.jpg" 
                   value="https://i.ibb.co/v4ykcTp4/e8283822a24d72765a6f51a801e33525-1-1-2.png">
        </div>
        
        <button onclick="generateAnimation()" id="generateBtn">
            Generate Mouth Animation
        </button>
        
        <div id="result" class="result">
            <div id="resultContent"></div>
        </div>
        
        <div style="margin-top: 30px; padding: 20px; background: #f0f0f0; border-radius: 8px;">
            <h3>How it works:</h3>
            <ul>
                <li>Upload image URL or use the demo image</li>
                <li>OpenCV detects the mouth area</li>
                <li>Adds pulsing, glowing and sparkle animations</li>
                <li>Generates MP4 video file</li>
            </ul>
        </div>
    </div>

    <script>
        async function generateAnimation() {
            const imageUrl = document.getElementById('imageUrl').value;
            const resultDiv = document.getElementById('result');
            const resultContent = document.getElementById('resultContent');
            const generateBtn = document.getElementById('generateBtn');
            
            if (!imageUrl) {
                showResult('Please enter a valid image URL!', 'error');
                return;
            }
            
            // Show loading
            generateBtn.disabled = true;
            generateBtn.innerHTML = '<span class="spinner"></span> Processing...';
            showResult('<div class="loading"><span class="spinner"></span> Creating animation... This may take 30-60 seconds.</div>', '');
            
            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ image_url: imageUrl })
                });
                
                if (response.ok) {
                    const blob = await response.blob();
                    const videoUrl = URL.createObjectURL(blob);
                    
                    showResult(`
                        <div class="success">
                            <h3>✅ Animation Created Successfully!</h3>
                            <video controls autoplay loop>
                                <source src="${videoUrl}" type="video/mp4">
                            </video>
                            <br>
                            <a href="${videoUrl}" download="mouth_animation.mp4" class="download-link">
                                📥 Download MP4
                            </a>
                        </div>
                    `, 'success');
                } else {
                    const error = await response.text();
                    showResult(`Error: ${error}`, 'error');
                }
            } catch (error) {
                showResult(`Network error: ${error.message}`, 'error');
            } finally {
                generateBtn.disabled = false;
                generateBtn.innerHTML = 'Generate Mouth Animation';
            }
        }
        
        function showResult(content, type) {
            const resultDiv = document.getElementById('result');
            const resultContent = document.getElementById('resultContent');
            
            resultContent.innerHTML = content;
            resultDiv.className = `result ${type}`;
            resultDiv.style.display = 'block';
        }
    </script>
</body>
</html>
"""

def download_image(url, filename):
    """Download image from URL"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response:
            with open(filename, 'wb') as f:
                f.write(response.read())
        return True
    except Exception as e:
        print(f"Error downloading image: {e}")
        return False

def detect_mouth(image):
    """Detect mouth using OpenCV Haar Cascade"""
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y + h//2:y + h, x:x + w]
            mouths = mouth_cascade.detectMultiScale(roi_gray, 1.8, 20)
            
            if len(mouths) > 0:
                mouth = max(mouths, key=lambda m: m[2] * m[3])
                mx, my, mw, mh = mouth
                return (x + mx, y + h//2 + my, mw, mh)
        
        return None
    except Exception as e:
        print(f"Error in mouth detection: {e}")
        return None

def create_mouth_animation(image_path, output_path, duration=4, fps=15):
    """Create animated video with mouth effects"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False
        
        # Resize image if too large
        height, width = img.shape[:2]
        if width > 800:
            scale = 800 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height))
            height, width = new_height, new_width
        
        mouth_coords = detect_mouth(img)
        
        if mouth_coords is None:
            # If no mouth detected, add animation to center
            mouth_coords = (width//2 - 60, height//2 - 30, 120, 60)
        
        mx, my, mw, mh = mouth_coords
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        total_frames = duration * fps
        
        for frame_num in range(total_frames):
            frame = img.copy()
            t = frame_num / fps
            
            # Animation 1: Pulsing outline
            pulse = int(8 + 4 * math.sin(t * 3))
            color_intensity = int(255 * (0.5 + 0.5 * math.sin(t * 2)))
            outline_color = (0, color_intensity, 255 - color_intensity)
            
            cv2.rectangle(frame, 
                         (mx - pulse, my - pulse), 
                         (mx + mw + pulse, my + mh + pulse), 
                         outline_color, max(2, pulse//3))
            
            # Animation 2: Moving sparkles
            for i in range(4):
                angle = (t * 2 + i * math.pi / 2) % (2 * math.pi)
                radius = 25 + 8 * math.sin(t * 2 + i)
                
                sparkle_x = int(mx + mw//2 + radius * math.cos(angle))
                sparkle_y = int(my + mh//2 + radius * math.sin(angle))
                
                if 0 <= sparkle_x < width and 0 <= sparkle_y < height:
                    cv2.circle(frame, (sparkle_x, sparkle_y), 2, (255, 255, 0), -1)
                    cv2.circle(frame, (sparkle_x, sparkle_y), 4, (255, 255, 255), 1)
            
            # Animation 3: Glowing effect
            glow_intensity = 0.2 + 0.15 * math.sin(t * 1.5)
            overlay = frame.copy()
            cv2.rectangle(overlay, (mx, my), (mx + mw, my + mh), (100, 200, 255), -1)
            cv2.addWeighted(frame, 1 - glow_intensity, overlay, glow_intensity, 0, frame)
            
            out.write(frame)
        
        out.release()
        return True
        
    except Exception as e:
        print(f"Error creating animation: {e}")
        return False

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/generate', methods=['POST'])
def generate_animation():
    try:
        data = request.get_json()
        image_url = data.get('image_url')
        
        if not image_url:
            return "No image URL provided", 400
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_input:
            input_path = temp_input.name
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_output:
            output_path = temp_output.name
        
        try:
            # Download image
            if not download_image(image_url, input_path):
                return "Failed to download image", 400
            
            # Create animation
            if not create_mouth_animation(input_path, output_path):
                return "Failed to create animation", 500
            
            # Return video file
            return send_file(output_path, 
                           mimetype='video/mp4',
                           as_attachment=True,
                           download_name='mouth_animation.mp4')
        
        finally:
            # Cleanup
            try:
                if os.path.exists(input_path):
                    os.unlink(input_path)
            except:
                pass
    
    except Exception as e:
        return f"Server error: {str(e)}", 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
