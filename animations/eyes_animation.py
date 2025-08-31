import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

def blink_eyes(image, frame_num, total_frames):
    h, w, _ = image.shape
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return image
    
    for face_landmarks in results.multi_face_landmarks:
        # Left and right eye landmark indices
        LEFT_EYE = [33, 160, 158, 133, 153, 144]
        RIGHT_EYE = [362, 385, 387, 263, 373, 380]

        for eye in [LEFT_EYE, RIGHT_EYE]:
            points = [(int(face_landmarks.landmark[i].x * w),
                       int(face_landmarks.landmark[i].y * h)) for i in eye]

            # Eye bounding box
            x_min = min([p[0] for p in points])
            x_max = max([p[0] for p in points])
            y_min = min([p[1] for p in points])
            y_max = max([p[1] for p in points])

            eye_roi = image[y_min:y_max, x_min:x_max].copy()

            # Blink effect (eye closes in middle frames)
            blink_ratio = abs(np.sin(np.pi * frame_num / total_frames))
            shrink = int((y_max - y_min) * blink_ratio * 0.6)

            if shrink > 0:
                eye_roi = cv2.resize(eye_roi, (x_max - x_min, (y_max - y_min) - shrink))
                # Pad back to original size
                eye_roi = cv2.copyMakeBorder(
                    eye_roi, shrink//2, shrink - shrink//2, 0, 0,
                    cv2.BORDER_REPLICATE
                )

            image[y_min:y_max, x_min:x_max] = eye_roi

    return image
