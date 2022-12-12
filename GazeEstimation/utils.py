import logging
import operator
import pathlib

import cv2
import yaml
from omegaconf import DictConfig

from common.face_model import FaceModel
from common.face_model_68 import FaceModel68
from common.face_model_mediapipe import FaceModelMediaPipe

logger = logging.getLogger(__name__)


def get_3d_face_model(config: DictConfig) -> FaceModel:
    if config.face_detector.mode == 'mediapipe':
        return FaceModelMediaPipe()
    else:
        return FaceModel68()


def generate_dummy_camera_params(config: DictConfig) -> None:
    logger.debug('Called _generate_dummy_camera_params()')
    if config.demo.image_path:
        path = pathlib.Path(config.demo.image_path).expanduser()
        image = cv2.imread(path.as_posix())
        h, w = image.shape[:2]
    elif config.demo.video_path:
        logger.debug(f'Open video {config.demo.video_path}')
        path = pathlib.Path(config.demo.video_path).expanduser().as_posix()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f'{config.demo.video_path} is not opened.')
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()
    else:
        raise ValueError
    logger.debug(f'Frame size is ({w}, {h})')
    logger.debug(f'Close video {config.demo.video_path}')
    logger.debug(f'Create a dummy camera param file tmp/camera_params.yaml')
    dic = {
        'image_width': w,
        'image_height': h,
        'camera_matrix': {
            'rows': 3,
            'cols': 3,
            'data': [w, 0., w // 2, 0., w, h // 2, 0., 0., 1.]
        },
        'distortion_coefficients': {
            'rows': 1,
            'cols': 5,
            'data': [0., 0., 0., 0., 0.]
        }
    }
    with open('tmp/camera_params.yaml', 'w') as f:
        yaml.safe_dump(dic, f)
    config.gaze_estimator.camera_params = 'tmp/camera_params.yaml'
    logger.debug(
        'Update config.gaze_estimator.camera_params to tmp/camera_params.yaml'
    )


def _expanduser(path: str) -> str:
    if not path:
        return path
    return pathlib.Path(path).expanduser().as_posix()


def expanduser_all(config: DictConfig) -> None:
    config.gaze_estimator.checkpoint = _expanduser(
        config.gaze_estimator.checkpoint)
    config.gaze_estimator.camera_params = _expanduser(
        config.gaze_estimator.camera_params)
    config.gaze_estimator.normalized_camera_params = _expanduser(
        config.gaze_estimator.normalized_camera_params)
    if hasattr(config.demo, 'image_path'):
        config.demo.image_path = _expanduser(config.demo.image_path)
    if hasattr(config.demo, 'video_path'):
        config.demo.video_path = _expanduser(config.demo.video_path)
    if hasattr(config.demo, 'output_dir'):
        config.demo.output_dir = _expanduser(config.demo.output_dir)


def expanduser_env_all(env_config: DictConfig) -> None:
    env_config.MASTER_DOMAIN_PREFIX = _expanduser(
        env_config.MASTER_DOMAIN_PREFIX)


def _check_path(config: DictConfig, key: str) -> None:
    path_str = operator.attrgetter(key)(config)
    path = pathlib.Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f'config.{key}: {path.as_posix()} not found.')
    if not path.is_file():
        raise ValueError(f'config.{key}: {path.as_posix()} is not a file.')


def check_path_all(config: DictConfig) -> None:
    _check_path(config, 'gaze_estimator.checkpoint')
    _check_path(config, 'gaze_estimator.camera_params')
    _check_path(config, 'gaze_estimator.normalized_camera_params')
    if config.demo.image_path:
        _check_path(config, 'demo.image_path')
    if config.demo.video_path:
        _check_path(config, 'demo.video_path')