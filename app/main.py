from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil, uuid, requests
import cv2
import mediapipe as mp
from .pipeline import process_no_crop

app = FastAPI(title="FOMM Mask Animation API")

BASE = Path(__file__).parent
STATIC = BASE / "static"
OUT = STATIC / "outputs"
WEIGHTS = BASE / "weights"

app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

@app.get("/")
def root():
    return {"status": "ok", "endpoints": ["/animate", "/animate-url"]}

@app.post("/animate")
async def animate(source_image: UploadFile = File(...),
                  driving_video: UploadFile = File(...),
                  fps: int = 25):
    job_id = uuid.uuid4().hex[:10]
    workdir = OUT / job_id

    try:
        if workdir.exists() and not workdir.is_dir():
            workdir.unlink()
        workdir.mkdir(parents=True, exist_ok=True)

        src = workdir / "source.jpg"
        drv = workdir / "driving.mp4"
        outp = workdir / "result.mp4"

        with src.open("wb") as f:
            shutil.copyfileobj(source_image.file, f)
        with drv.open("wb") as f:
            shutil.copyfileobj(driving_video.file, f)

        result = process_no_crop(src, drv, WEIGHTS, outp, fps=fps)

        if not outp.exists():
            return {"error": "Animation failed: result.mp4 not created", "job_id": job_id}

        url = f"/static/outputs/{job_id}/result.mp4"
        return {"url": url, "job_id": job_id}

    except Exception as e:
        return {"error": str(e), "job_id": job_id}


@app.post("/animate-url")
async def animate_from_url(image_url: str = Form(...),
                           driving_video: UploadFile = File(...),
                           fps: int = 25):
    job_id = uuid.uuid4().hex[:10]
    workdir = OUT / job_id

    try:
        if workdir.exists() and not workdir.is_dir():
            workdir.unlink()
        workdir.mkdir(parents=True, exist_ok=True)

        src = workdir / "source.jpg"
        drv = workdir / "driving.mp4"
        outp = workdir / "result.mp4"

        # Step 1: Download image
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            with src.open("wb") as f:
                f.write(response.content)
        except Exception as e:
            return {"error": f"Image download failed: {str(e)}", "job_id": job_id}

        # Step 2: MediaPipe face detection
        try:
            image = cv2.imread(str(src))
            if image is None:
                return {"error": "Downloaded image is unreadable", "job_id": job_id}

            mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)
            results = mp_face.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    for pt in face_landmarks.landmark:
                        x = int(pt.x * image.shape[1])
                        y = int(pt.y * image.shape[0])
                        cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
                cv2.imwrite(str(src), image)
        except Exception as e:
            return {"error": f"MediaPipe processing failed: {str(e)}", "job_id": job_id}

        # Step 3: Save driving video
        try:
            with drv.open("wb") as f:
                shutil.copyfileobj(driving_video.file, f)
        except Exception as e:
            return {"error": f"Driving video save failed: {str(e)}", "job_id": job_id}

        # Step 4: Animate
        try:
            result = process_no_crop(src, drv, WEIGHTS, outp, fps=fps)
            if not outp.exists():
                return {"error": "Animation failed: result.mp4 not created", "job_id": job_id}
        except Exception as e:
            return {"error": f"FOMM animation failed: {str(e)}", "job_id": job_id}

        url = f"/static/outputs/{job_id}/result.mp4"
        return {"url": url, "job_id": job_id}

    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}", "job_id": job_id}
