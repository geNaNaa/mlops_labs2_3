import bentoml
from bentoml.io import Image as BentoImage, JSON
import numpy as np
import onnxruntime as ort
from PIL import Image as PILImage, ImageDraw
import base64
import io
import os

# Определяем сервис
svc = bentoml.Service("bee_wasp_detector")

# Загрузка модели при инициализации модуля
model_path = "/content/best.onnx"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Файл модели не найден по пути {model_path}")

session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
class_names = ['Bee', 'Wasp']
@svc.api(input=BentoImage(), output=JSON())
def predict(input_image: PILImage.Image) -> dict:
    # Предобработка изображения
    image = input_image.convert("RGB")
    image_resized = image.resize((256, 256))
    image_data = np.array(image_resized).astype('float32') / 255.0
    image_data = image_data.transpose(2, 0, 1)
    image_data = np.expand_dims(image_data, axis=0)

    # Отладочные выводы для проверки предобработки
    print(f"Image resized shape: {image_resized.size}")
    print(f"Image data shape: {image_data.shape}")
    print(f"Image data stats: min={image_data.min()}, max={image_data.max()}, mean={image_data.mean()}")

    # Выполнение инференса
    outputs = session.run([output_name], {input_name: image_data})

    # Отладочные выводы для проверки выходных данных модели
    print(f"Outputs shape: {np.array(outputs).shape}")
    print(f"First 5 detections: {outputs[0][:5]}")
    print(f"Total detections: {len(outputs[0])}")

    # Постобработка и рисование результатов
    result_image, detected_classes = postprocess(outputs, image_resized)

    # Конвертация изображения в base64
    buffered = io.BytesIO()
    result_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Получение имен классов из class_ids
    detected_class_names = [class_names[class_id] for class_id in detected_classes]

    return {"image_base64": img_str, "detected_classes": detected_class_names}

def postprocess(outputs, image):
    # Обработка выходных данных модели и рисование bounding boxes
    detections = outputs[0]
    detections = np.squeeze(detections)

    # Транспонируем, если необходимо
    if detections.shape[0] == 6:
        detections = detections.T  # Преобразуем в форму (N, 6)

    print(f"Количество детекций: {len(detections)}")
    print(f"Содержимое detections: {detections[:5]}")

    # Порог уверенности
    conf_threshold = 0.1

    draw = ImageDraw.Draw(image)

    width, height = image.size

    detected_classes = []

    for detection in detections:
        x1, y1, x2, y2, conf, class_prob = detection[:6]

        # Отладочный вывод
        print(f"Raw detection: x1={x1}, y1={y1}, x2={x2}, y2={y2}, conf={conf}, class_prob={class_prob}")

        # Проверяем уверенность
        if conf < conf_threshold:
            continue

        # Определяем класс на основе вероятности
        class_id = 1 if class_prob > 0.5 else 0  # Применяем порог 0.5

        if class_id < 0 or class_id >= len(class_names):
            continue  # Пропускаем, если class_id некорректен

        # Масштабируем координаты к размеру изображения, если они уже в пикселях, то не масштабируем
        # Если модель возвращает координаты в пикселях, закомментируйте следующие строки:
        # x1 *= width
        # y1 *= height
        # x2 *= width
        # y2 *= height

        # Преобразуем координаты в целые числа
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # Убедимся, что x1 <= x2 и y1 <= y2
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        # Ограничиваем координаты размерами изображения
        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))

        label = f"{class_names[class_id]}: {conf:.2f}"
        print(f"Обнаружен объект: {label} с координатами ({x1}, {y1}), ({x2}, {y2})")

        detected_classes.append(class_id)

        # Рисуем bounding box и метку
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1 - 10), label, fill="red")

    return image, detected_classes