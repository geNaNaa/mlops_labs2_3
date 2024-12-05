import bentoml
from bentoml.io import Image
import numpy as np
import cv2
from ultralytics import YOLO

@bentoml.service()
class CarDetectionService:
    def __init__(self):
        self.model = YOLO("models/best.onnx")

    def predict(self, image: np.ndarray):
        results = self.model.predict(image)
        return results

    def draw_detections(self, image: np.ndarray, detections: list) -> np.ndarray:
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            label = f"Car: {conf:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return image

    @bentoml.api(input=Image(), output=Image())
    def detect(self, image: np.ndarray) -> np.ndarray:
        results = self.predict(image)
        return self.draw_detections(image, results)