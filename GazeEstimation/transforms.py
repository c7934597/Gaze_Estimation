from typing import Any

import cv2
import albumentations as A
import torchvision.transforms as T
from omegaconf import DictConfig
from albumentations.pytorch.transforms import ToTensorV2


def create_transform(config: DictConfig) -> Any:
    return _create_gaze_transform(config)


def create_time_interval_transform() -> Any:
    return _create_gaze_time_interval_transform()


def _create_gaze_transform(config: DictConfig) -> Any:
    size = tuple(config.gaze_estimator.image_size)
    transform = T.Compose([
        T.Lambda(lambda x: cv2.resize(x, size)),
        T.Lambda(lambda x: x[:, :, ::-1].copy()),  # BGR -> RGB
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,
                                                     0.225]),  # RGB
    ])
    return transform


def _create_gaze_time_interval_transform() -> Any:
    return A.Compose([A.Normalize(
        mean = [0.485, 0.456, 0.406], 
        std = [0.229, 0.224, 0.225], 
        max_pixel_value = [255.0], 
        p = 1.0), # 正規化。
            ToTensorV2(p = 1.0) # 歸一化
            ], p = 1.0)