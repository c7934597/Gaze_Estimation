import timm
import logging

from typing import List
from omegaconf import DictConfig
from common import Camera, Face, FacePartsName, HeadPoseNormalizer, LandmarkEstimator
from transforms import create_transform, create_time_interval_transform
from utils import get_3d_face_model

import torch
import numpy as np
import torch.nn as nn

from torchvision import models

logger = logging.getLogger(__name__)


class build_model(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = timm.create_model("tf_efficientnetv2_b0", pretrained = None)

        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.5), 
            nn.Linear(n_features, 2, bias = True)
        )

    def forward(self, x):
        x = self.model(x)
        return x

class GazeEstimator:
    EYE_KEYS = [FacePartsName.REYE, FacePartsName.LEYE]

    def __init__(self, config: DictConfig):
        self._config = config

        self._face_model3d = get_3d_face_model(config)

        self.camera = Camera(config.gaze_estimator.camera_params)
        self._normalized_camera = Camera(
            config.gaze_estimator.normalized_camera_params)

        self._landmark_estimator = LandmarkEstimator(config)
        self._head_pose_normalizer = HeadPoseNormalizer(
            self.camera, self._normalized_camera,
            self._config.gaze_estimator.normalized_camera_distance)
        logger.info('Gaze model load start')
        self._gaze_estimation_model = self._load_gaze_model()
        logger.info('Gaze model load end')
        logger.info('Gaze time interval model start')
        self._gaze_estimation_time_interval_model = self._load_gaze_time_interval_model()
        logger.info('Gaze time interval model end')
        self._transform = create_transform(config)
        self._time_interval_transform = create_time_interval_transform()

    def _load_gaze_model(self) -> torch.nn.Module:
        model = torch.load("models/gaze.pt")
        model.to(torch.device(self._config.device))
        model.eval()
        return model

    def _load_gaze_time_interval_model(self) -> torch.nn.Module:
        model = build_model()
        model.load_state_dict(torch.load('models/efficientnet_v2_b0.pth'))
        model.to(torch.device(self._config.device))
        model.eval()
        return model

    def detect_faces(self, image: np.ndarray) -> List[Face]:
        return self._landmark_estimator.detect_faces(image)

    def estimate_gaze(self, image: np.ndarray, face: Face) -> None:
        self._face_model3d.estimate_head_pose(face, self.camera)
        self._face_model3d.compute_3d_pose(face)
        self._face_model3d.compute_face_eye_centers(face, self._config.mode)
        
        self._head_pose_normalizer.normalize(image, face)
        self._run_gaze_model(face)


    @torch.no_grad()
    def _run_gaze_model(self, face: Face) -> None:
        image = self._transform(face.normalized_image).unsqueeze(0)

        device = torch.device(self._config.device)
        image = image.to(device)
        prediction = self._gaze_estimation_model(image)
        prediction = prediction.cpu().numpy()

        face.normalized_gaze_angles = prediction[0]
        face.angle_to_vector()
        face.denormalize_gaze_vector()


    @torch.no_grad()
    def _run_gaze_time_interval_model(self, image) -> None:
        image = self._time_interval_transform(image = image)['image'].unsqueeze(0)

        device = torch.device(self._config.device)
        image = image.to(device)
        prediction = self._gaze_estimation_time_interval_model(image)
        percentage = torch.nn.functional.softmax(prediction, dim=1).cpu().detach().numpy()[0] * 100
        return percentage
