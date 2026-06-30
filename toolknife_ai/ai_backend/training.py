from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from config.settings import (
    DEFAULT_IMAGE_SIZE,
    DETECTION_WEIGHT,
    KEYPOINT_WEIGHT,
    ROOT,
    TRAINING_RUN_DIR,
    ensure_directories,
)


@dataclass(frozen=True)
class TrainingRun:
    run_id: str
    model: str
    status: str
    dataset_yaml: str
    epochs: int
    imgsz: int
    log_file: str
    metadata_file: str


def create_training_run(
    model: str,
    dataset_yaml: str,
    epochs: int = 50,
    imgsz: int = DEFAULT_IMAGE_SIZE,
) -> TrainingRun:
    ensure_directories()
    if model not in {"detection", "keypoint"}:
        raise ValueError("model must be detection or keypoint")

    dataset_path = Path(dataset_yaml)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset YAML was not found: {dataset_yaml}")

    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{model}-{uuid4().hex[:8]}"
    run_dir = TRAINING_RUN_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "train.log"
    metadata_file = run_dir / "metadata.json"
    base_weight = DETECTION_WEIGHT if model == "detection" else KEYPOINT_WEIGHT

    cmd = [
        sys.executable,
        "-m",
        "ai_backend.train_worker",
        "--model-weight",
        str(base_weight),
        "--dataset-yaml",
        str(dataset_path),
        "--epochs",
        str(epochs),
        "--imgsz",
        str(imgsz),
        "--run-dir",
        str(run_dir),
    ]

    run = TrainingRun(
        run_id=run_id,
        model=model,
        status="running",
        dataset_yaml=str(dataset_path),
        epochs=epochs,
        imgsz=imgsz,
        log_file=str(log_file),
        metadata_file=str(metadata_file),
    )
    metadata = asdict(run) | {"command": cmd}
    metadata_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    with log_file.open("ab") as log:
        subprocess.Popen(cmd, stdout=log, stderr=log, cwd=ROOT)

    return run


def read_training_run(run_id: str) -> dict[str, object]:
    metadata = TRAINING_RUN_DIR / run_id / "metadata.json"
    if not metadata.exists():
        raise FileNotFoundError(f"Training run was not found: {run_id}")
    return json.loads(metadata.read_text(encoding="utf-8"))


def list_training_runs() -> list[dict[str, object]]:
    ensure_directories()
    runs = []
    for metadata in sorted(TRAINING_RUN_DIR.glob("*/metadata.json"), reverse=True):
        runs.append(json.loads(metadata.read_text(encoding="utf-8")))
    return runs
