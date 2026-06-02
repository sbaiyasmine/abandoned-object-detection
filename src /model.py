import onnxruntime as ort
import numpy as np


def load_model(model_path: str):
    return ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"]
    )


def run_inference(session, input_data: np.ndarray):
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: input_data})
