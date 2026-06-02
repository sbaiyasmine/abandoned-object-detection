import cv2
import numpy as np
import onnxruntime as ort
import paho.mqtt.client as mqtt
import json
import time
import csv
import psutil
from datetime import datetime, timezone
from pathlib import Path
from picamera2 import Picamera2


class BaseVisionSystem:

    def __init__(self, config_path="config.json"):
        with open(config_path) as f:
            self.config = json.load(f)

        self.session = ort.InferenceSession(
            self.config["model_path"],
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_disconnect = self._on_disconnect
        self.mqtt_client.connect(
            self.config["mqtt_broker"],
            self.config["mqtt_port"],
            keepalive=60
        )
        self.mqtt_client.loop_start()

        self.picam2 = Picamera2()
        cam_config = self.picam2.create_preview_configuration(
            main={"size": (320, 320), "format": "RGB888"}
        )
        self.picam2.configure(cam_config)
        self.picam2.start()
        time.sleep(1)

        print("Camera Picamera2 ouverte : True")

        self.log_file = f"logs/equipe_{self.config['equipe_id']}_events.csv"
        Path("logs").mkdir(exist_ok=True)
        self._init_csv_log()

        self.frame_id = 0
        self.fps = 0.0

    def _on_disconnect(self, client, userdata, rc):
        max_retries = 10
        for attempt in range(max_retries):
            try:
                client.reconnect()
                return
            except Exception as exc:
                print(f"[MQTT] Tentative reconnexion {attempt + 1}/{max_retries} échouée: {exc}")
                time.sleep(5)
        print("[MQTT] Impossible de se reconnecter après plusieurs tentatives.")

    def _init_csv_log(self):
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'equipe_id', 'event_type',
                'confidence', 'alert', 'alert_level', 'alert_message'
            ])

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def postprocess(self, outputs, frame: np.ndarray):
        raise NotImplementedError

    def publish_event(self, event_type: str, confidence: float,
                      bbox: list, metadata: dict,
                      alert: bool = False,
                      alert_level: str = "none",
                      alert_message: str = ""):
        msg = {
            "equipe_id": self.config["equipe_id"],
            "equipe_name": self.config["equipe_name"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "camera_id": self.config["camera_id"],
            "frame_id": self.frame_id,
            "event_type": event_type,
            "confidence": float(confidence),
            "bbox": bbox,
            "metadata": metadata,
            "alert": alert,
            "alert_level": alert_level,
            "alert_message": alert_message
        }

        equipe_id = self.config['equipe_id']
        topic = f"surveillance/equipe_{equipe_id}/events"
        if alert:
            topic = f"surveillance/equipe_{equipe_id}/alert"

        self.mqtt_client.publish(topic, json.dumps(msg), qos=1)
        print("[MQTT EVENT]", topic, msg)

        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                msg['timestamp'], equipe_id, event_type,
                confidence, alert, alert_level, alert_message
            ])

    def publish_heartbeat(self):
        try:
            temp = self.get_cpu_temp()
            cpu = psutil.cpu_percent()
        except Exception:
            temp, cpu = 0, 0

        msg = {
            "equipe_id": self.config["equipe_id"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "fps": round(self.fps, 1),
            "cpu_usage": cpu,
            "temperature": temp
        }

        equipe_id = self.config['equipe_id']
        topic = f"surveillance/equipe_{equipe_id}/heartbeat"
        self.mqtt_client.publish(topic, json.dumps(msg), qos=0)
        print("[MQTT]", topic, msg)

    def get_cpu_temp(self) -> float:
        try:
            with open("/sys/class/thermal/thermal_zone0/temp") as f:
                return int(f.read()) / 1000.0
        except Exception:
            return 0.0

    def run(self):
        last_heartbeat = time.time()
        fps_counter = 0
        fps_start = time.time()

        print(f"[Équipe {self.config['equipe_id']}] Démarrage pipeline...")

        try:
            while True:
                frame_rgb = self.picam2.capture_array()
                frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                self.frame_id += 1

                inputs = self.preprocess(frame)
                outputs = self.session.run(None, {self.input_name: inputs})
                self.postprocess(outputs, frame)

                fps_counter += 1
                elapsed = time.time() - fps_start

                if elapsed >= 1.0:
                    self.fps = fps_counter / elapsed
                    print("FPS:", round(self.fps, 1))
                    fps_counter = 0
                    fps_start = time.time()

                if time.time() - last_heartbeat >= 5.0:
                    self.publish_heartbeat()
                    last_heartbeat = time.time()

        except KeyboardInterrupt:
            print(f"[Équipe {self.config['equipe_id']}] Arrêt demandé.")

        finally:
            self.picam2.stop()
            self.mqtt_client.loop_stop()
