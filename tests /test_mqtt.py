import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def publisher():
    with patch("paho.mqtt.client.Client") as MockMQTT:
        mock_client = MagicMock()
        MockMQTT.return_value = mock_client

        from src.mqtt_publisher import MQTTPublisher

        pub = MQTTPublisher(
            broker="127.0.0.1",
            port=1883,
            equipe_id=6
        )
        pub.client = mock_client

    return pub


def test_base_topic_format():
    from src.mqtt_publisher import BASE_TOPIC
    assert BASE_TOPIC == "surveillance/equipe_06"


def test_mqtt_connection_init():
    with patch("paho.mqtt.client.Client") as MockMQTT:
        mock_client = MagicMock()
        MockMQTT.return_value = mock_client

        from src.mqtt_publisher import MQTTPublisher

        MQTTPublisher(
            broker="127.0.0.1",
            port=1883,
            equipe_id=6
        )

        mock_client.connect.assert_called_once_with(
            "127.0.0.1",
            1883,
            keepalive=60
        )


def test_heartbeat_format(publisher):
    published = {}

    def capture(topic, payload, **kwargs):
        published["topic"] = topic
        published["payload"] = json.loads(payload)

    publisher.client.publish.side_effect = capture

    publisher.publish_heartbeat(
        fps=12.5,
        cpu_usage=45.0,
        temperature=52.0
    )

    assert published["payload"]["equipe_id"] == 6
    assert published["payload"]["status"] == "running"
    assert "fps" in published["payload"]


def test_heartbeat_topic(publisher):
    publisher.client.publish.return_value = None
    publisher.publish_heartbeat()

    call_args = publisher.client.publish.call_args
    assert "heartbeat" in call_args[0][0]


def test_event_with_object_class(publisher):
    published = {}

    def capture(topic, payload, **kwargs):
        published["topic"] = topic
        published["payload"] = json.loads(payload)

    publisher.client.publish.side_effect = capture

    publisher.publish_event(
        event_type="abandoned_object",
        confidence=0.85,
        bbox=[10, 20, 100, 120],
        metadata={
            "object_class": "suitcase",
            "abandoned_since_seconds": 8
        },
        alert=True,
        alert_level="high",
        alert_message="suitcase abandoned"
    )

    assert "alert" in published["topic"]
    assert published["payload"]["event_type"] == "abandoned_object"
    assert published["payload"]["metadata"]["object_class"] == "suitcase"
    assert published["payload"]["alert"] is True
