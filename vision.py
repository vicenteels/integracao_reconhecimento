"""Adaptação sem efeitos de importação de reconhecimento_service.py (YuNet/SFace)."""
from pathlib import Path

import cv2
import numpy as np


class Vision:
    def __init__(self, model_dir):
        folder = Path(model_dir)
        detector = folder / "face_detection_yunet_2023mar.onnx"
        recognizer = folder / "face_recognition_sface_2021dec.onnx"
        if not detector.is_file() or not recognizer.is_file():
            raise FileNotFoundError("Modelos YuNet/SFace não encontrados em MODEL_DIR.")
        self.detector = cv2.FaceDetectorYN.create(str(detector), "", (320, 320), score_threshold=0.85, nms_threshold=0.3, top_k=5000)
        self.recognizer = cv2.FaceRecognizerSF.create(str(recognizer), "")

    def faces(self, frame):
        height, width = frame.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(frame)
        return [] if faces is None else list(faces)

    def embedding(self, frame, face):
        aligned = self.recognizer.alignCrop(frame, face)
        vector = self.recognizer.feature(aligned).ravel().astype(np.float32)
        norm = np.linalg.norm(vector)
        if not np.isfinite(norm) or norm <= 0 or vector.size != 128:
            raise ValueError("SFace produziu embedding inválido.")
        return (vector / norm).tolist()
