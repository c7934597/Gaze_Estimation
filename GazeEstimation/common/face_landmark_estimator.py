import logging
from typing import List

import torch
import face_alignment
import mediapipe
import numpy as np
from omegaconf import DictConfig

from .face import Face

logger = logging.getLogger(__name__)

class LandmarkEstimator:
    def __init__(self, config: DictConfig):
        self.mode = config.face_detector.mode
        if self.mode == 'face_alignment_sfd':
            logger.info('FD model load start')
            self.detector = torch.load("models/s3fd.pt", map_location=torch.device(config.device))
            logger.info('FD model load end')
            logger.info('FA model load start')
            self.predictor = face_alignment.FaceAlignment(
                face_alignment.LandmarksType._2D,
                flip_input=False,
                device=config.device)
            logger.info('FA model load end')
        elif self.mode == 'mediapipe':
            self.detector = mediapipe.solutions.face_mesh.FaceMesh(
                max_num_faces=config.face_detector.mediapipe_max_num_faces)
        else:
            raise ValueError

    def detect_faces(self, image: np.ndarray) -> List[Face]:
        if self.mode == 'face_alignment_sfd':
            return self._detect_faces_face_alignment_sfd(image)
        elif self.mode == 'mediapipe':
            return self._detect_faces_mediapipe(image)
        else:
            raise ValueError

    def _detect_faces_face_alignment_sfd(self,
                                         image: np.ndarray) -> List[Face]:
        bboxes = self.detector.detect_from_image(image[:, :, ::-1].copy())
        # print(bboxes) # [array([min_x, min_y, max_x, max_y, score])]
        bboxes = [bbox[:4] for bbox in bboxes]
        predictions = self.predictor.get_landmarks(image[:, :, ::-1],
                                                   detected_faces=bboxes)
        if predictions is None:
            predictions = []
        detected = []
        for bbox, landmarks in zip(bboxes, predictions):
            bbox = np.array(bbox, dtype=np.float).reshape(2, 2)
            detected.append(Face(bbox, landmarks))
        return detected

    def _detect_faces_mediapipe(self, image: np.ndarray) -> List[Face]:
        h, w = image.shape[:2]
        predictions = self.detector.process(image[:, :, ::-1])
        detected = []
        if predictions.multi_face_landmarks:
            for prediction in predictions.multi_face_landmarks:
                pts = np.array([(pt.x * w, pt.y * h)
                                for pt in prediction.landmark],
                               dtype=np.float64)
                bbox = np.vstack([pts.min(axis=0), pts.max(axis=0)])
                bbox = np.round(bbox).astype(np.int32)
                detected.append(Face(bbox, pts))
        return detected
