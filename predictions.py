from __future__ import annotations
import bentoml
from bentoml.io import Image, JSON
import cv2
import numpy as np

from ultralytics import YOLO

# Экземпляр BentoML сервиса
@bentoml.service(
    resources={"cpu": "2", "nvidia.com/gpu": "1"},  # Указать GPU, если доступен
    traffic={"timeout": 10},
)
class CarDetectionService:
    def __init__(self) -> None:
        # Загрузка YOLO-модели
        self.model = YOLO("")  # Укажите путь к модели

    @bentoml.api(input=Image(), output=Image())
    def detect_cars(self, image: np.ndarray) -> np.ndarray:
        # Предобработка изображения
        input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Преобразуем в RGB
        results = self.model.predict(input_image)  # Инференс с использованием YOLO

        # Постобработка: рендерим детекцию на изображении
        output_image = self.draw_detections(image, results)
        return output_image

    def draw_detections(self, image: np.ndarray, detections) -> np.ndarray:
        """
        Рисуем детекции на изображении
        """
        for det in detections:
            x1, y1, x2, y2, conf, cls = det  # координаты, вероятность, класс
            label = f"Car: {conf:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return image