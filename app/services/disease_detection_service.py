import io
import os
import cv2
import numpy as np
import base64
from PIL import Image
import onnxruntime as ort
from app.services.doctor_service import get_treatment_text

# =============================
# Model Paths and Initial Setup
# =============================
BASE_DIR = os.path.dirname(__file__)
yolo_model_path = os.path.join(BASE_DIR, "..", "models", "yolo.onnx")
mobilenet_model_path = os.path.join(BASE_DIR, "..", "models", "class.onnx")

# Load ONNX models
yolo_session = ort.InferenceSession(yolo_model_path)
mobilenet_session = ort.InferenceSession(mobilenet_model_path)

# Class names for MobileNet predictions
custom_class_names = {
    0: 'Early Blight', 1: 'Healthy', 2: 'Late Blight', 3: 'Leaf Miner',
    4: 'Leaf Mold', 5: 'Mosaic Virus', 6: 'Septoria',
    7: 'Spider Mites', 8: 'Yellow Leaf Curl Virus'
}

# =============================
# Helper Function for MobileNet
# =============================
def preprocess_for_mobilenet(img: np.ndarray) -> np.ndarray:
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 127.5 - 1.0
    img = np.expand_dims(img, axis=0)
    return img

# =============================
# Main Image Processing Function
# =============================
async def process_image(file):
    # Read and decode image
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Prepare image for YOLO
    input_img = cv2.resize(image_rgb, (640, 640))
    input_img = input_img.astype(np.float32) / 255.0
    input_img = np.transpose(input_img, (2, 0, 1))
    input_img = np.expand_dims(input_img, axis=0)
    input_img = np.ascontiguousarray(input_img)

    # Run YOLO detection
    yolo_inputs = {yolo_session.get_inputs()[0].name: input_img}
    yolo_outputs = yolo_session.run(None, yolo_inputs)
    detections = yolo_outputs[0][0]  # [num_detections, 6]

    detected_diseases = set()

    # Loop over detections
    for det in detections:
        if det.shape[0] != 6:
            continue
        x1, y1, x2, y2, score, class_id = map(float, det)
        if score < 0.3:
            continue

        # Crop infected region
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        crop = image_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Classify using MobileNet
        input_mobilenet = preprocess_for_mobilenet(crop)
        mobilenet_inputs = {mobilenet_session.get_inputs()[0].name: input_mobilenet}
        mobilenet_outputs = mobilenet_session.run(None, mobilenet_inputs)
        pred_idx = np.argmax(mobilenet_outputs[0], axis=-1).item()
        disease = custom_class_names.get(pred_idx, "Unknown")
        detected_diseases.add(disease)

        # Draw bounding box and label
        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_rgb, disease, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

    # Convert final image to base64
    pil_img = Image.fromarray(image_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    img_url = f"data:image/png;base64,{img_base64}"

    # Get treatment text using LLM
    if detected_diseases:
        diseases_list = list(detected_diseases)
        diseases_str = ", ".join(diseases_list)
        treatment = await get_treatment_text(diseases_str)
    else:
        diseases_list = []
        treatment = "No diseases detected in the image. The plant appears healthy."

    return {
        "image_url": img_url,
        "diseases": diseases_list,
        "treatment": treatment
    }
