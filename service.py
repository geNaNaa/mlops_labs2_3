import bentoml
from bentoml.io import Image, JSON
from PIL import Image as PILImage, ImageDraw
import numpy as np
import onnxruntime as ort

# Load the ONNX model
model_path = "/content/best.onnx"  # Replace with your ONNX model path
session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

# Define a preprocessing function
def preprocess(image: PILImage.Image):
    image = image.resize((640, 640))  # Resize to model's input size
    image = np.array(image).astype(np.float32)
    image = np.transpose(image, (2, 0, 1))  # Convert to CHW format
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image /= 255.0  # Normalize to [0, 1]
    return image

# Define a postprocessing function
def postprocess(image: PILImage.Image, outputs, conf_threshold=0.5):
    boxes, scores, labels = outputs
    draw = ImageDraw.Draw(image)

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, w, h = box
        y2 = y1 + h
        x2 = x1 + w
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1 - 10), f"Class {label}: {score:.2f}", fill="red")

    return image

# Define the inference function
def infer(image: PILImage.Image):
    input_tensor = preprocess(image)
    outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})
    outputs = np.transpose(np.squeeze(outputs[0]))
    rows = outputs.shape[0]
    boxes = []
    scores = []
    class_ids = []

    # x_factor = self.img_width / self.input_width
    # y_factor = self.img_height / self.input_height

    for i in range(rows):
        classes_scores = outputs[i][4:]
        max_score = np.amax(classes_scores)
        x_factor = 676/640
        y_factor = 380/640
        if max_score >= 0.5:
            class_id = np.argmax(classes_scores)
            x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
            left = int((x - w / 2) * x_factor)
            top = int((y - h / 2) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)
            print(x, y, w, h)
            class_ids.append(class_id)
            scores.append(max_score)
            boxes.append([left, top, width, height])
    return postprocess(image, (boxes, scores, class_ids))

# Define the BentoML service
svc = bentoml.Service("car_detection_service")

@svc.api(input=Image(), output=Image())
def detect(image: PILImage.Image) -> PILImage.Image:
    return infer(image)