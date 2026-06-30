from __future__ import annotations

import base64
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from ai_backend.serializers import boxes_to_json, keypoints_to_json
from config.settings import (
    DEFAULT_CONFIDENCE,
    DETECTION_WEIGHT,
    KEYPOINT_WEIGHT,
    LOG_DIR,
    OUTPUT_DIR,
    UPLOAD_DIR,
    ensure_directories,
    ensure_weight,
)


@dataclass(frozen=True)
class ModelResult:
    job_id: str
    model: str
    status: str
    input_image: str
    annotated_image: str
    result_json: str
    detections: list[dict[str, Any]]
    keypoints: list[list[dict[str, float | int]]]
    annotated_image_base64: str


class DetectionRuntime:
    def __init__(self) -> None:
        ensure_weight(DETECTION_WEIGHT, "detection")
        from ultralytics import YOLO

        self.model = YOLO(str(DETECTION_WEIGHT))

    def predict(self, image_path: Path, original_filename: str | None = None) -> ModelResult:
        ensure_directories()
        stored = store_upload(image_path, original_filename)
        result = self.model.predict(source=str(stored), conf=DEFAULT_CONFIDENCE, save=False, verbose=False)[0]
        stem = safe_name(Path(original_filename or stored.name).stem)
        job_id = f"{stem}-detection-{uuid4().hex[:8]}"
        out_dir = OUTPUT_DIR / "det"
        out_dir.mkdir(parents=True, exist_ok=True)
        annotated = out_dir / f"{job_id}_annotated.jpg"
        detections = boxes_to_json(result)
        save_bbox_image(stored, detections, annotated)
        payload = {
            "job_id": job_id,
            "model": "detection",
            "status": "success",
            "weight_path": str(DETECTION_WEIGHT),
            "input_image": str(stored),
            "annotated_image": str(annotated),
            "detections": detections,
            "keypoints": [],
        }
        result_json = out_dir / f"{job_id}_result.json"
        result_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        write_log(job_id, "detection", stored, annotated)
        return result_from_payload(payload, result_json)


class KeypointRuntime:
    def __init__(self) -> None:
        ensure_weight(KEYPOINT_WEIGHT, "keypoint")
        from ultralytics import YOLO

        self.model = YOLO(str(KEYPOINT_WEIGHT))

    def predict(self, image_path: Path, original_filename: str | None = None) -> ModelResult:
        ensure_directories()
        stored = store_upload(image_path, original_filename)
        result = self.model.predict(source=str(stored), conf=DEFAULT_CONFIDENCE, save=False, verbose=False)[0]
        stem = safe_name(Path(original_filename or stored.name).stem)
        job_id = f"{stem}-keypoint-{uuid4().hex[:8]}"
        out_dir = OUTPUT_DIR / "keypoint"
        out_dir.mkdir(parents=True, exist_ok=True)
        annotated = out_dir / f"{job_id}_annotated.jpg"
        save_plotted_image(result.plot(), annotated)
        payload = {
            "job_id": job_id,
            "model": "keypoint",
            "status": "success",
            "weight_path": str(KEYPOINT_WEIGHT),
            "input_image": str(stored),
            "annotated_image": str(annotated),
            "detections": boxes_to_json(result),
            "keypoints": keypoints_to_json(result),
        }
        result_json = out_dir / f"{job_id}_result.json"
        result_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        write_log(job_id, "keypoint", stored, annotated)
        return result_from_payload(payload, result_json)


_detection_runtime: DetectionRuntime | None = None
_keypoint_runtime: KeypointRuntime | None = None


def get_detection_runtime() -> DetectionRuntime:
    global _detection_runtime
    if _detection_runtime is None:
        _detection_runtime = DetectionRuntime()
    return _detection_runtime


def get_keypoint_runtime() -> KeypointRuntime:
    global _keypoint_runtime
    if _keypoint_runtime is None:
        _keypoint_runtime = KeypointRuntime()
    return _keypoint_runtime


def store_upload(image_path: Path, original_filename: str | None = None) -> Path:
    ensure_directories()
    source_name = original_filename or image_path.name
    suffix = (Path(source_name).suffix or image_path.suffix or ".jpg").lower()
    stem = safe_name(Path(source_name).stem)
    target = UPLOAD_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{stem}-{uuid4().hex[:8]}{suffix}"
    shutil.copy2(image_path, target)
    return target


def save_plotted_image(image_array: Any, path: Path) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image_array)


def save_bbox_image(
    image_path: Path,
    detections: list[dict[str, Any]],
    output_path: Path,
) -> None:
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not open image for bbox drawing: {image_path}")

    for detection in detections:
        try:
            x1, y1, x2, y2 = [
                int(round(float(value)))
                for value in detection["xyxy"]
            ]
        except (KeyError, TypeError, ValueError):
            continue

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        class_name = str(detection.get("class_name") or "object")
        confidence = detection.get("confidence")
        label = class_name
        if confidence is not None:
            label = f"{class_name} {float(confidence):.2f}"
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise ValueError(f"Could not save bbox image: {output_path}")


def image_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def result_from_payload(payload: dict[str, Any], result_json: Path) -> ModelResult:
    annotated = Path(str(payload["annotated_image"]))
    return ModelResult(
        job_id=str(payload["job_id"]),
        model=str(payload["model"]),
        status=str(payload["status"]),
        input_image=str(payload["input_image"]),
        annotated_image=str(annotated),
        result_json=str(result_json),
        detections=list(payload.get("detections", [])),
        keypoints=list(payload.get("keypoints", [])),
        annotated_image_base64=image_to_base64(annotated),
    )


def write_log(job_id: str, model: str, source: Path, annotated: Path) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        f"job_id={job_id}",
        f"model={model}",
        f"input_image={source}",
        f"annotated_image={annotated}",
        "status=success",
    ]
    (LOG_DIR / f"{job_id}.log").write_text("\n".join(lines), encoding="utf-8")


def safe_name(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value).strip().strip(".")
    return cleaned or f"image-{uuid4().hex[:8]}"
