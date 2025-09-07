import sys
import yaml
from tqdm.auto import tqdm

import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch

from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from animate import normalize_kp

import ffmpeg
from os.path import splitext
from shutil import copyfileobj
from tempfile import NamedTemporaryFile
from pathlib import Path

# ✅ Ensure Python version is compatible
if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7+")

def load_checkpoints(config_path, checkpoint_path, cpu=False):
    """
    Load pretrained weights and model configuration
    """
    with open(config_path) as f:
        config = yaml.full_load(f)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])

    if cpu or not torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        generator.cuda()
        kp_detector.cuda()
        checkpoint = torch.load(checkpoint_path)

    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])

    generator.eval()
    kp_detector.eval()

    return generator, kp_detector

def make_animation(source_image, driving_video, generator, kp_detector,
                   relative=True, adapt_movement_scale=True, cpu=False):
    """
    Generate animation frames from source image and driving video
    """
    with torch.no_grad():
        predictions = []

        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)

        if not cpu and torch.cuda.is_available():
            source = source.cuda()
            driving = driving.cuda()

        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])

        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            if not cpu and torch.cuda.is_available():
                driving_frame = driving_frame.cuda()

            kp_driving = kp_detector(driving_frame)
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial,
                                   use_relative_movement=relative,
                                   use_relative_jacobian=relative,
                                   adapt_movement_scale=adapt_movement_scale)

            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)
            prediction = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]
            predictions.append(prediction)

    return predictions
