print("modification salma moufid")
import json
import paho.mqtt.client as mqtt
from datetime import datetime, timezone


BASE_TOPIC = "surveillance/equipe_06"


class MQTTPublisher:
    """Standalone MQTT publisher pour l'équipe 06."""

    def __init__(self, broker: str = "192.168.1.100", port: int = 1883, equipe_id: int = 6):
        self.equipe_id = equipe_id
        self.client = mqtt.Client()
        self.client.connect(
            broker,
            port,
            keepalive=60
        )
        self.client.loop_start()

    def publish_event(self, event_type: str, confidence: float,
                      bbox: list, metadata: dict,
                      alert: bool = False,
                      alert_level: str = "none",
                      alert_message: str = "") -> None:
        msg = {
            "equipe_id": self.equipe_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "confidence": float(confidence),
            "bbox": bbox,
            "metadata": metadata,
            "alert": alert,
            "alert_level": alert_level,
            "alert_message": alert_message,
        }

        topic = f"{BASE_TOPIC}/events"
        if alert:
            topic = f"{BASE_TOPIC}/alert"

        self.client.publish(topic, json.dumps(msg), qos=1)

    def publish_heartbeat(self, fps: float = 0.0, cpu_usage: float = 0.0,
                          temperature: float = 0.0) -> None:
        msg = {
            "equipe_id": self.equipe_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "fps": round(fps, 1),
            "cpu_usage": cpu_usage,
            "temperature": temperature,
        }

        self.client.publish(
            f"{BASE_TOPIC}/heartbeat",
            json.dumps(msg),
            qos=0
        )

    def stop(self) -> None:
        self.client.loop_stop()
        self.client.disconnect()
