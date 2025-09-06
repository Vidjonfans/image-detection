import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from moviepy.editor import ImageSequenceClip

# FOMM imports
import sys
sys.path.append(str(Path(__file__).parent / "fomm"))
from demo import load_checkpoints, make_animation
from omegaconf import OmegaConf

mp_face_mesh = mp.solutions.face_mesh

# Landmark indices
EYE_IDXS_LEFT = [33, 133, 145, 159, 160, 144, 153, 154]
EYE_IDXS_RIGHT = [362, 263, 374, 386, 385, 380, 373, 390]
LIPS_IDXS = [0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 87, 88, 91,
             308, 310, 311, 312, 317, 318, 321, 324]

def _prep_for_fomm(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (256, 256))

def _read_video_frames(path: Path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()
    if not frames:
        raise ValueError("No frames in driving video")
    return frames

def _get_landmarks(image_bgr):
    with mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True, max_num_faces=1) as fm:
        res = fm.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        raise ValueError("No face detected")
    return res.multi_face_landmarks[0]

def _pts_from_idxs(landmarks, w, h, idxs):
    return np.array([(int(landmarks.landmark[i].x * w),
                      int(landmarks.landmark[i].y * h)) for i in idxs], dtype=np.int32)

def _make_mask(h, w, regions):
    mask = np.zeros((h, w), dtype=np.uint8)
    for r in regions:
        if len(r) >= 3:
            cv2.fillConvexPoly(mask, r, 255)
    mask = cv2.GaussianBlur(mask, (15, 15), 5)
    return mask

def run_fomm_fullface(source_bgr, driving_frames_bgr, weights_dir: Path, cpu=True):
    source_256 = _prep_for_fomm(source_bgr)
    driving_256 = [_prep_for_fomm(f) for f in driving_frames_bgr]

    cfg_path = weights_dir / "vox-256.yaml"
    ckpt_path = weights_dir / "vox-cpk.pth.tar"
    if not cfg_path.exists() or not ckpt_path.exists():
        raise FileNotFoundError("FOMM weights not found in app/weights (vox-256.yaml, vox-cpk.pth.tar)")

    config = OmegaConf.load(str(cfg_path))
    generator, kp_detector = load_checkpoints(
        config_path=str(cfg_path),
        checkpoint_path=str(ckpt_path),
        cpu=cpu
    )

    anim_rgb_list = make_animation(
        source_256, driving_256, generator, kp_detector,
        relative=True, adapt_movement_scale=True
    )  # list of RGB 256x256

    # Resize back to source size and convert to BGR
    H, W = source_bgr.shape[:2]
    out_bgr = [cv2.cvtColor(cv2.resize(f, (W, H)), cv2.COLOR_RGB2BGR) for f in anim_rgb_list]
    return out_bgr

def composite_masked(source_bgr, anim_bgr, landmarks):
    H, W = source_bgr.shape[:2]
    left_eye = _pts_from_idxs(landmarks, W, H, EYE_IDXS_LEFT)
    right_eye = _pts_from_idxs(landmarks, W, H, EYE_IDXS_RIGHT)
    lips = _pts_from_idxs(landmarks, W, H, LIPS_IDXS)

    mask = _make_mask(H, W, [left_eye, right_eye, lips])
    center = (W // 2, H // 2)
    # SeamlessClone expects 3-channel mask
    mask3 = cv2.merge([mask, mask, mask])
    blended = cv2.seamlessClone(anim_bgr, source_bgr, mask, center, cv2.MIXED_CLONE)
    return blended

def images_to_mp4(frames_bgr, out_path: Path, fps=25):
    rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
    clip = ImageSequenceClip(rgb_frames, fps=fps)
    clip.write_videofile(str(out_path), codec="libx264", audio=False, fps=fps, bitrate="2000k")

def process_no_crop(source_img_path: Path, driving_video_path: Path, weights_dir: Path, out_path: Path, fps=25):
    source = cv2.imread(str(source_img_path))
    if source is None:
        raise ValueError("Cannot read source image")

    driving_frames = _read_video_frames(driving_video_path)
    landmarks = _get_landmarks(source)

    anim_frames = run_fomm_fullface(source, driving_frames, weights_dir, cpu=True)

    final_frames = [composite_masked(source, af, landmarks) for af in anim_frames]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    images_to_mp4(final_frames, out_path, fps=fps)
    return out_path
