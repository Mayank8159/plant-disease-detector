import os
import threading
from typing import List, Tuple

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    # Windows often lacks tflite-runtime wheels; use TensorFlow Lite interpreter as fallback.
    from tensorflow.lite.python.interpreter import Interpreter


MODEL_PATH = os.getenv("MODEL_PATH", "model.tflite")
LABELS_PATH = os.getenv("LABELS_PATH", "labels.txt")
IMAGE_SIZE: Tuple[int, int] = (224, 224)
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.70"))

app = FastAPI(title="Plant Disease Detector API")

_interpreter: Interpreter | None = None
_input_details = None
_output_details = None
_labels: List[str] | None = None
_model_lock = threading.Lock()


def _load_labels() -> List[str]:
    if not os.path.exists(LABELS_PATH):
        return []

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _get_interpreter() -> Interpreter:
    global _interpreter, _input_details, _output_details, _labels

    if _interpreter is None:
        with _model_lock:
            if _interpreter is None:
                if not os.path.exists(MODEL_PATH):
                    raise RuntimeError(f"TFLite model not found at: {MODEL_PATH}")

                interpreter = Interpreter(model_path=MODEL_PATH, num_threads=1)
                interpreter.allocate_tensors()

                _interpreter = interpreter
                _input_details = interpreter.get_input_details()
                _output_details = interpreter.get_output_details()
                _labels = _load_labels()

    return _interpreter


def _preprocess_image(upload: UploadFile) -> np.ndarray:
    try:
        upload.file.seek(0)
        with Image.open(upload.file) as img:
            img = img.convert("RGB")
            img = img.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
            arr = np.asarray(img, dtype=np.float32)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc

    arr /= 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def _prepare_input_for_model(image_batch: np.ndarray, input_info: dict) -> np.ndarray:
    expected_dtype = input_info["dtype"]
    if expected_dtype == np.float32:
        return image_batch.astype(np.float32, copy=False)

    scale, zero_point = input_info.get("quantization", (0.0, 0))
    if scale and scale > 0:
        quantized = np.round((image_batch / scale) + zero_point)
    else:
        quantized = np.round(image_batch * 255.0)

    if expected_dtype == np.uint8:
        return np.clip(quantized, 0, 255).astype(np.uint8, copy=False)

    if expected_dtype == np.int8:
        return np.clip(quantized, -128, 127).astype(np.int8, copy=False)

    return image_batch.astype(expected_dtype, copy=False)


def _infer(image_batch: np.ndarray) -> Tuple[int, float, np.ndarray]:
    interpreter = _get_interpreter()
    input_info = _input_details[0]
    output_info = _output_details[0]

    model_input = _prepare_input_for_model(image_batch, input_info)

    with _model_lock:
        interpreter.set_tensor(input_info["index"], model_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_info["index"])

    scores = np.squeeze(output).astype(np.float32, copy=False)
    out_scale, out_zero_point = output_info.get("quantization", (0.0, 0))
    if out_scale and out_scale > 0:
        scores = (scores - float(out_zero_point)) * float(out_scale)

    if scores.ndim != 1:
        scores = scores.reshape(-1)

    if scores.max() > 1.0 or scores.min() < 0.0:
        # If the model output is logits, convert to probabilities.
        shifted = scores - np.max(scores)
        exp_scores = np.exp(shifted)
        denom = np.sum(exp_scores)
        if denom > 0:
            scores = exp_scores / denom

    top_idx = int(np.argmax(scores))
    top_conf = float(scores[top_idx])

    return top_idx, top_conf, scores


def _resolve_label(index: int) -> str:
    if _labels and 0 <= index < len(_labels):
        return _labels[index]
    return f"class_{index}"


@app.on_event("startup")
def _startup() -> None:
    _get_interpreter()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    image_batch = _preprocess_image(file)

    try:
        top_idx, confidence, _ = _infer(image_batch)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Inference failed") from exc

    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "prediction": "Inconclusive",
            "confidence": confidence,
            "threshold": CONFIDENCE_THRESHOLD,
        }

    return {
        "prediction": _resolve_label(top_idx),
        "confidence": confidence,
        "threshold": CONFIDENCE_THRESHOLD,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
