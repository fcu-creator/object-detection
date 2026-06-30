from __future__ import annotations

from typing import Any


def boxes_to_json(result: Any) -> list[dict[str, Any]]:
    if result is None or result.boxes is None:
        return []

    names = result.names
    boxes = result.boxes
    output = []
    for index in range(len(boxes)):
        cls_id = int(boxes.cls[index].item()) if boxes.cls is not None else -1
        confidence = None
        if boxes.conf is not None:
            confidence = float(boxes.conf[index].item())

        output.append(
            {
                "class_id": cls_id,
                "class_name": names.get(cls_id, str(cls_id)),
                "confidence": confidence,
                "xyxy": [
                    float(value)
                    for value in boxes.xyxy[index].detach().cpu().tolist()
                ],
            }
        )
    return output


def keypoints_to_json(result: Any) -> list[list[dict[str, float | int]]]:
    if result is None or result.keypoints is None or result.keypoints.xy is None:
        return []

    xy = result.keypoints.xy.detach().cpu().tolist()
    confidence = None
    if getattr(result.keypoints, "conf", None) is not None:
        confidence = result.keypoints.conf.detach().cpu().tolist()

    all_instances = []
    for instance_index, points in enumerate(xy):
        instance_points = []
        for point_index, point in enumerate(points):
            item: dict[str, float | int] = {
                "index": point_index,
                "x": float(point[0]),
                "y": float(point[1]),
            }
            if confidence is not None:
                item["confidence"] = float(confidence[instance_index][point_index])
            instance_points.append(item)
        all_instances.append(instance_points)
    return all_instances

