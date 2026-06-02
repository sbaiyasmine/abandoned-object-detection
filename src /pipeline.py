import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import time

from common.base_vision import BaseVisionSystem


CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush"
]


class AbandonedObjectSystem(BaseVisionSystem):

    def __init__(self, config_path="config.json"):
        super().__init__(config_path)

        self.objects = {}

        self.min_area = self.config["custom_params"].get(
            "min_motion_area",
            500
        )

    def preprocess(self, frame):
        img = cv2.resize(frame, (320, 320))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)
        img = img.astype(np.float32) / 255.0
        return img[np.newaxis, ...]

    def detect_yolo(self, outputs, frame):
        predictions = outputs[0][0].transpose()

        h0, w0 = frame.shape[:2]

        sx = w0 / 320
        sy = h0 / 320

        detections = []

        for det in predictions:
            x, y, w, h = det[:4]
            scores = det[4:]

            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])

            if confidence < self.config["confidence_threshold"]:
                continue

            if class_id >= len(CLASSES):
                continue

            x1 = int((x - w / 2) * sx)
            y1 = int((y - h / 2) * sy)
            x2 = int((x + w / 2) * sx)
            y2 = int((y + h / 2) * sy)

            detections.append({
                "label": CLASSES[class_id],
                "confidence": confidence,
                "bbox": [x1, y1, x2 - x1, y2 - y1]
            })

        return detections

    def postprocess(self, outputs, frame):
        detections = self.detect_yolo(outputs, frame)

        print("YOLO detections:", detections[:3])

        threshold_time = self.config["alert_thresholds"]["abandoned_seconds"]

        now = time.time()

        if not detections:
            return

        best = max(detections, key=lambda d: d["confidence"])

        x, y, w, h = best["bbox"]
        object_label = best["label"]
        confidence = best["confidence"]

        key = object_label

        if key not in self.objects:
            self.objects[key] = now

        elapsed = now - self.objects[key]

        alert = elapsed >= threshold_time

        color = (0, 0, 255) if alert else (0, 255, 0)

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            color,
            2
        )

        cv2.putText(
            frame,
            f"{object_label} {confidence:.2f}",
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

        self.publish_event(
            event_type="abandoned_object" if alert else "object_present",
            confidence=confidence,
            bbox=[x, y, w, h],
            metadata={
                "object_class": object_label,
                "abandoned_since_seconds": round(elapsed, 1)
            },
            alert=alert,
            alert_level="high" if alert else "none",
            alert_message=f"{object_label} abandoned" if alert else ""
        )
