import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import json
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def dummy_frame():
    return np.zeros((720, 1280, 3), dtype=np.uint8)


@pytest.fixture
def dummy_config(tmp_path):
    cfg = {
        "equipe_id": 6,
        "equipe_name": "detection_objets_abandonnes",
        "camera_id": "cam_06",
        "mqtt_broker": "127.0.0.1",
        "mqtt_port": 1883,
        "stream_fps": 15,
        "model_path": "models/yolov8n_finetuned.onnx",
        "confidence_threshold": 0.5,
        "alert_thresholds": {"abandoned_seconds": 5},
        "custom_params": {"min_motion_area": 700},
    }

    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(cfg))
    return str(config_file)


def make_pipeline(config_path):
    with patch("onnxruntime.InferenceSession"), \
         patch("paho.mqtt.client.Client") as MockMQTT, \
         patch("cv2.VideoCapture"):

        mock_mqtt = MagicMock()
        MockMQTT.return_value = mock_mqtt

        from src.pipeline import AbandonedObjectSystem

        sys_obj = AbandonedObjectSystem(config_path)
        sys_obj.mqtt_client = mock_mqtt

    return sys_obj


def test_preprocess_shape(dummy_frame, dummy_config):
    sys_obj = make_pipeline(dummy_config)
    result = sys_obj.preprocess(dummy_frame)
    assert result.shape == (1, 3, 640, 640)


def test_preprocess_dtype(dummy_frame, dummy_config):
    sys_obj = make_pipeline(dummy_config)
    result = sys_obj.preprocess(dummy_frame)
    assert result.dtype == np.float32


def test_preprocess_normalization(dummy_config):
    frame = np.ones((720, 1280, 3), dtype=np.uint8) * 255
    sys_obj = make_pipeline(dummy_config)
    result = sys_obj.preprocess(frame)
    assert result.max() <= 1.0
    assert result.min() >= 0.0


def test_publish_event_format(dummy_config):
    sys_obj = make_pipeline(dummy_config)
    sys_obj.mqtt_client = MagicMock()

    published = {}

    def capture_publish(topic, payload, **kwargs):
        published["topic"] = topic
        published["payload"] = json.loads(payload)

    sys_obj.mqtt_client.publish.side_effect = capture_publish

    with patch("builtins.open", MagicMock()):
        sys_obj.publish_event(
            event_type="abandoned_object",
            confidence=0.9,
            bbox=[0, 0, 100, 100],
            metadata={
                "object_class": "backpack",
                "abandoned_since_seconds": 5
            },
            alert=True,
            alert_level="high",
            alert_message="backpack abandoned"
        )

    assert "payload" in published
    assert published["payload"]["equipe_id"] == 6
    assert published["payload"]["event_type"] == "abandoned_object"
    assert published["payload"]["metadata"]["object_class"] == "backpack"
    assert published["payload"]["alert"] is True
    assert "alert" in published["topic"]
