import time
import cv2
import numpy as np
import onnxruntime as ort
from picamera2 import Picamera2

MODEL_PATH = "models/yolov8n_finetuned_320.onnx"
IMG_SIZE = 320
TOTAL_FRAMES = 100
INFER_EVERY = 2

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name


def preprocess(frame):
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)
    img = img.astype(np.float32) / 255.0
    return img[np.newaxis, ...]


picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"size": (320, 320), "format": "RGB888"}
)

picam2.configure(config)
picam2.start()

time.sleep(1)

for _ in range(5):
    frame_rgb = picam2.capture_array()
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    inp = preprocess(frame)
    session.run(None, {input_name: inp})

t0 = time.time()

for i in range(TOTAL_FRAMES):
    frame_rgb = picam2.capture_array()
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    if i % INFER_EVERY == 0:
        inp = preprocess(frame)
        session.run(None, {input_name: inp})

t1 = time.time()

picam2.stop()

fps_system = TOTAL_FRAMES / (t1 - t0)

print("Total frames:", TOTAL_FRAMES)
print("Inference every:", INFER_EVERY)    
print("FPS système caméra:", round(fps_system, 2))

if fps_system > 8:
    print("OK: FPS système caméra > 8")
else:
    print("FPS caméra encore insuffisant")
