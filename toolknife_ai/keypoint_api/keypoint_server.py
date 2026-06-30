from __future__ import annotations

import base64
import tempfile
from pathlib import Path
from typing import Any

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from ai_backend.model_service import get_keypoint_runtime
from config.settings import ALLOWED_SUFFIXES, KEYPOINT_PORT, MAX_UPLOAD_BYTES

app = FastAPI(title="ToolKnife Find Point API", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "service": "toolknife-find-point-api", "port": KEYPOINT_PORT}


@app.post("/api/keypoint")
async def keypoint(file: UploadFile = File(...)) -> dict[str, object]:
    tmp_path = await upload_to_temp(file)
    try:
        result = get_keypoint_runtime().predict(tmp_path, file.filename)
        return result.__dict__
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/api/keypoint-focus")
async def keypoint_focus(
    file: UploadFile = File(...),
    focus_scale: float = Form(default=0.65),
) -> dict[str, object]:
    if focus_scale <= 0:
        raise HTTPException(status_code=400, detail="focus_scale must be greater than 0.")
    tmp_path = await upload_to_temp(file)
    try:
        result = get_keypoint_runtime().predict(tmp_path, file.filename)
        payload = result.__dict__
        payload["focus_scale"] = focus_scale
        payload["keypoint_image_base64"] = payload.get("annotated_image_base64", "")
        payload["focused_image_base64"] = focus_crop_by_best_keypoint(tmp_path, payload, focus_scale)
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


def focus_crop_by_best_keypoint(image_path: Path, keypoint: dict[str, object], scale: float) -> str:
    image = cv2.imread(str(image_path))
    if image is None:
        raise HTTPException(status_code=400, detail="Uploaded image could not be opened for focus cropping.")

    selected = select_keypoint(keypoint)
    height, width = image.shape[:2]
    center_x = float(selected["x"])
    center_y = float(selected["y"])
    side = max(32.0, min(width, height) * scale)

    crop_x1 = max(0, int(round(center_x - side / 2)))
    crop_y1 = max(0, int(round(center_y - side / 2)))
    crop_x2 = min(width, int(round(center_x + side / 2)))
    crop_y2 = min(height, int(round(center_y + side / 2)))
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        raise HTTPException(status_code=422, detail="Focus crop box is empty.")

    focused = image[crop_y1:crop_y2, crop_x1:crop_x2]
    ok, encoded = cv2.imencode(".jpg", focused)
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode focused image.")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def select_keypoint(keypoint: dict[str, object]) -> dict[str, float | int]:
    points = [
        point
        for instance in list(keypoint.get("keypoints") or [])
        if isinstance(instance, list)
        for point in instance
        if isinstance(point, dict) and "x" in point and "y" in point
    ]
    if not points:
        raise HTTPException(status_code=422, detail="No keypoint was returned, so no focus crop can be created.")
    selected: dict[str, Any] = max(points, key=lambda item: float(item.get("confidence") or 0.0))
    return {
        "index": int(selected.get("index") or 0),
        "x": float(selected["x"]),
        "y": float(selected["y"]),
        "confidence": float(selected.get("confidence") or 0.0),
    }
