from __future__ import annotations

import base64
import tempfile
from pathlib import Path

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from ai_backend.model_service import get_detection_runtime
from config.settings import ALLOWED_SUFFIXES, DETECTION_PORT, MAX_UPLOAD_BYTES

app = FastAPI(title="ToolKnife Find Tool API", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "service": "toolknife-find-tool-api", "port": DETECTION_PORT}


@app.post("/api/detect")
async def detect(file: UploadFile = File(...)) -> dict[str, object]:
    tmp_path = await upload_to_temp(file)
    try:
        result = get_detection_runtime().predict(tmp_path, file.filename)
        return result.__dict__
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/api/detect-crop")
async def detect_crop(
    file: UploadFile = File(...),
    crop_scale: float = Form(default=1.5),
) -> dict[str, object]:
    if crop_scale <= 0:
        raise HTTPException(status_code=400, detail="crop_scale must be greater than 0.")
    tmp_path = await upload_to_temp(file)
    try:
        result = get_detection_runtime().predict(tmp_path, file.filename)
        payload = result.__dict__
        payload["detections"] = select_best_detection(payload)
        payload["annotated_image_base64"] = draw_single_detection(tmp_path, payload["detections"])
        payload["crop_scale"] = crop_scale
        payload["cropped_image_base64"] = crop_by_best_box(tmp_path, payload, crop_scale)
        payload["detection_image_base64"] = payload.get("annotated_image_base64", "")
        return payload
    finally:
        tmp_path.unlink(missing_ok=True)


async def upload_to_temp(file: UploadFile) -> Path:
    suffix = Path(file.filename or "upload.jpg").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail="Only jpg, jpeg, png, bmp, and webp images are supported.")
    body = await file.read()
    if len(body) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Upload is too large.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(body)
        return Path(tmp.name)


def crop_by_best_box(image_path: Path, detection: dict[str, object], scale: float) -> str:
    image = cv2.imread(str(image_path))
    if image is None:
        raise HTTPException(status_code=400, detail="Uploaded image could not be opened for cropping.")

    detections = list(detection.get("detections") or [])
    if not detections:
        raise HTTPException(status_code=422, detail="No detection box was returned, so no crop can be created.")

    selected = max(detections, key=lambda item: float(item.get("confidence") or 0.0))
    try:
        x1, y1, x2, y2 = [float(value) for value in selected["xyxy"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="Selected detection does not contain a valid xyxy box.") from exc

    height, width = image.shape[:2]
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    crop_w = box_w * scale
    crop_h = box_h * scale

    crop_x1 = max(0, int(round(center_x - crop_w / 2)))
    crop_y1 = max(0, int(round(center_y - crop_h / 2)))
    crop_x2 = min(width, int(round(center_x + crop_w / 2)))
    crop_y2 = min(height, int(round(center_y + crop_h / 2)))
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        raise HTTPException(status_code=422, detail="Expanded crop box is empty.")

    ok, encoded = cv2.imencode(".jpg", image[crop_y1:crop_y2, crop_x1:crop_x2])
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode cropped image.")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def select_best_detection(detection: dict[str, object]) -> list[dict[str, object]]:
    detections = list(detection.get("detections") or [])
    if not detections:
        raise HTTPException(status_code=422, detail="No detection box was returned, so no crop can be created.")
    return [max(detections, key=lambda item: float(item.get("confidence") or 0.0))]


def draw_single_detection(image_path: Path, detections: list[dict[str, object]]) -> str:
    image = cv2.imread(str(image_path))
    if image is None:
        raise HTTPException(status_code=400, detail="Uploaded image could not be opened for drawing.")

    for detection in detections:
        try:
            x1, y1, x2, y2 = [int(round(float(value))) for value in detection["xyxy"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="Selected detection does not contain a valid xyxy box.") from exc
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = str(detection.get("class_name") or "tool")
        confidence = detection.get("confidence")
        if confidence is not None:
            label = f"{label} {float(confidence):.2f}"
        cv2.putText(
            image,
            label,
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode detection image.")
    return base64.b64encode(encoded.tobytes()).decode("ascii")
