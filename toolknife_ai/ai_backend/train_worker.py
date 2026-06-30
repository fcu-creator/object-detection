from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a ToolKnife YOLO training job.")
    parser.add_argument("--model-weight", required=True)
    parser.add_argument("--dataset-yaml", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    from ultralytics import YOLO

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model_weight)
    model.train(
        data=args.dataset_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=str(run_dir),
        name="weights",
    )


if __name__ == "__main__":
    main()
