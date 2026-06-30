from __future__ import annotations

import base64
import json
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from config.settings import (
    ALLOWED_SUFFIXES,
    CROP_PORT,
    MAX_UPLOAD_BYTES,
    OUTPUT_DIR,
    ensure_directories,
)

app = FastAPI(title="ToolKnife Crop API", version="2.0")


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "service": "toolknife-crop-api", "port": CROP_PORT}


@app.post("/api/crop")
async def crop(
    file: UploadFile = File(...),
    detection_json: str = Form(...),
    crop_scale: float = Form(default=1.4),
) -> dict[str, object]:
    if crop_scale <= 0:
        raise HTTPException(status_code=400, detail="crop_scale must be greater than 0.")
    detection = parse_detection_json(detection_json)
    tmp_path = await upload_to_temp(file)
    try:
        return create_crop(tmp_path, file.filename or "upload.jpg", detection, crop_scale)
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/api/focus-crop")
async def focus_crop(
    file: UploadFile = File(...),
    keypoint_json: str = Form(...),
    focus_scale: float = Form(default=0.45),
) -> dict[str, object]:
    if focus_scale <= 0:
        raise HTTPException(status_code=400, detail="focus_scale must be greater than 0.")
    keypoint = parse_detection_json(keypoint_json)
    tmp_path = await upload_to_temp(file)
    try:
        return create_focus_crop(tmp_path, file.filename or "upload.jpg", keypoint, focus_scale)
    finally:
        tmp_path.unlink(missing_ok=True)


def parse_detection_json(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="detection_json is not valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail="detection_json must be a JSON object.")
    return parsed


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


def create_crop(image_path: Path, original_filename: str, detection: dict[str, Any], scale: float) -> dict[str, object]:
    ensure_directories()
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
    side = max(box_w, box_h) * scale

    crop_x1 = max(0, int(round(center_x - side / 2)))
    crop_y1 = max(0, int(round(center_y - side / 2)))
    crop_x2 = min(width, int(round(center_x + side / 2)))
    crop_y2 = min(height, int(round(center_y + side / 2)))
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        raise HTTPException(status_code=422, detail="Expanded crop box is empty.")

    out_dir = OUTPUT_DIR / "crop"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_name(Path(original_filename).stem)
    job_id = f"{stem}-crop-{uuid4().hex[:8]}"
    expanded_path = out_dir / f"{job_id}_expanded_box.jpg"
    crop_path = out_dir / f"{job_id}_crop.jpg"
    result_path = out_dir / f"{job_id}_result.json"

    expanded = image.copy()
    cv2.rectangle(expanded, (round_int(x1), round_int(y1)), (round_int(x2), round_int(y2)), (0, 0, 255), 2)
    cv2.rectangle(expanded, (crop_x1, crop_y1), (crop_x2, crop_y2), (0, 255, 255), 2)
    cv2.imwrite(str(expanded_path), expanded)
    cv2.imwrite(str(crop_path), image[crop_y1:crop_y2, crop_x1:crop_x2])

    payload: dict[str, object] = {
        "job_id": job_id,
        "model": "crop",
        "status": "success",
        "scale": scale,
        "source_detection_job_id": detection.get("job_id"),
        "selected_detection": selected,
        "original_box_xyxy": [x1, y1, x2, y2],
        "expanded_box_xyxy": [crop_x1, crop_y1, crop_x2, crop_y2],
        "expanded_box_image": str(expanded_path),
        "cropped_image": str(crop_path),
        "result_json": str(result_path),
        "expanded_box_image_base64": image_to_base64(expanded_path),
        "cropped_image_base64": image_to_base64(crop_path),
        "created_at": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
    }
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def create_focus_crop(image_path: Path, original_filename: str, keypoint: dict[str, Any], scale: float) -> dict[str, object]:
    ensure_directories()
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

    out_dir = OUTPUT_DIR / "crop"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_name(Path(original_filename).stem)
    job_id = f"{stem}-focus-{uuid4().hex[:8]}"
    marked_path = out_dir / f"{job_id}_marked.jpg"
    crop_path = out_dir / f"{job_id}_crop.jpg"
    result_path = out_dir / f"{job_id}_result.json"

    marked = image.copy()
    cv2.circle(marked, (round_int(center_x), round_int(center_y)), 5, (0, 0, 255), -1)
    cv2.rectangle(marked, (crop_x1, crop_y1), (crop_x2, crop_y2), (0, 255, 255), 2)
    cv2.imwrite(str(marked_path), marked)
    cv2.imwrite(str(crop_path), image[crop_y1:crop_y2, crop_x1:crop_x2])

    payload: dict[str, object] = {
        "job_id": job_id,
        "model": "focus-crop",
        "status": "success",
        "scale": scale,
        "source_keypoint_job_id": keypoint.get("job_id"),
        "selected_keypoint": selected,
        "focus_box_xyxy": [crop_x1, crop_y1, crop_x2, crop_y2],
        "marked_image": str(marked_path),
        "cropped_image": str(crop_path),
        "result_json": str(result_path),
        "marked_image_base64": image_to_base64(marked_path),
        "cropped_image_base64": image_to_base64(crop_path),
        "created_at": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
    }
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def select_keypoint(keypoint: dict[str, Any]) -> dict[str, float | int]:
    points = [
        point
        for instance in list(keypoint.get("keypoints") or [])
        if isinstance(instance, list)
        for point in instance
        if isinstance(point, dict) and "x" in point and "y" in point
    ]
    if not points:
        raise HTTPException(status_code=422, detail="No keypoint was returned, so no focus crop can be created.")
    selected = max(points, key=lambda item: float(item.get("confidence") or 0.0))
    return {
        "index": int(selected.get("index") or 0),
        "x": float(selected["x"]),
        "y": float(selected["y"]),
        "confidence": float(selected.get("confidence") or 0.0),
    }


def round_int(value: float) -> int:
    return int(round(value))


def image_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def safe_name(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value).strip().strip(".")
    return cleaned or f"image-{uuid4().hex[:8]}"
