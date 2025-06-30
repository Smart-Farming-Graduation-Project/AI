import cv2
import numpy as np
import base64
import io
import os
from PIL import Image
import onnxruntime as ort

# =======================================
# Load YOLO Model for Weed Detection
# =======================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "weed.onnx")
weed_session = ort.InferenceSession(MODEL_PATH)

# =======================================
# Preprocess Image for YOLO
# =======================================
def preprocess_image(img: np.ndarray) -> np.ndarray:
    img = cv2.resize(img, (640, 640))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    img = np.expand_dims(img, axis=0)
    return np.ascontiguousarray(img)

# =======================================
# Main Detection Function
# =======================================
async def detect_weed(file):
    # Load and decode image
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Step 2: Preprocess and inference
    input_tensor = preprocess_image(image_rgb)
    inputs = {weed_session.get_inputs()[0].name: input_tensor}
    output = weed_session.run(None, inputs)[0][0]

    # Analyze detections
    max_confidence = 0.0
    has_weed = False

    for det in output:
        if len(det) != 6:
            continue

        x1, y1, x2, y2, score, _ = map(float, det)
        if score < 0.4:
            continue

        has_weed = True
        max_confidence = max(max_confidence, score)

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image_rgb, "Weed", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Convert annotated image to base64
    pil_img = Image.fromarray(image_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    image_url = f"data:image/png;base64,{img_base64}"

    # Return structured response
    return {
        "image_url": image_url,
        "has_weed": has_weed,
        "confidence": round(float(max_confidence), 4) if has_weed else 0.0
    }
