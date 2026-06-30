from __future__ import annotations

import base64
import ipaddress
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
import re
import urllib.error
import urllib.request
from uuid import uuid4

from flask import Flask, jsonify, request, send_from_directory


BASE_DIR = Path(__file__).resolve().parent
PHOTO_DIR = BASE_DIR / "photos"
PHOTO_ORIGINAL_DIR = PHOTO_DIR / "original"
PHOTO_DETECTION_DIR = PHOTO_DIR / "object_detection"
PHOTO_KEYPOINT_DIR = PHOTO_DIR / "keypoint_detection"
PENDING_DIR = BASE_DIR / "pending"
RESULT_DIR = BASE_DIR / "results"
CERT_DIR = BASE_DIR / "certs"

for folder in (
    PHOTO_ORIGINAL_DIR,
    PHOTO_DETECTION_DIR,
    PHOTO_KEYPOINT_DIR,
    PENDING_DIR,
    RESULT_DIR,
    CERT_DIR,
):
    folder.mkdir(parents=True, exist_ok=True)

BASE_NAME_RE = re.compile(r"^tool_\d{8}_\d{6}_\d{3}$")
AI_DETECT_CROP_URL = "http://detection-api:8002/api/detect-crop"
AI_KEYPOINT_FOCUS_URL = "http://keypoint-api:8003/api/keypoint-focus"
CROP_SCALE = "1.5"
KEYPOINT_FOCUS_SCALE = "0.8"

app = Flask(__name__, static_folder="web", static_url_path="")


class InferenceStageError(RuntimeError):
    def __init__(self, stage: str, message: str, preview_image: str) -> None:
        super().__init__(message)
        self.stage = stage
        self.preview_image = preview_image


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.post("/api/captures")
def capture_photo():
    payload = request.get_json(silent=True) or {}
    try:
        image_bytes = decode_jpeg_data_url(payload.get("image", ""))
    except ValueError as error:
        return jsonify(error=str(error)), 400

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    base_name = f"tool_{timestamp}"
    raw_path = pending_raw_path(base_name)
    raw_path.write_bytes(image_bytes)
    shutil.copy2(raw_path, PHOTO_ORIGINAL_DIR / f"{base_name}.jpg")

    return jsonify(
        baseName=base_name,
        previewImage=f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('ascii')}",
        originalSavedAs=str(PHOTO_ORIGINAL_DIR / f"{base_name}.jpg"),
    )


@app.post("/api/infer")
def infer_photo():
    payload = request.get_json(silent=True) or {}
    base_name = payload.get("baseName", "")
    if not BASE_NAME_RE.fullmatch(base_name):
        return jsonify(error="Invalid base name"), 400

    raw_path = pending_raw_path(base_name)
    if not raw_path.exists():
        return jsonify(error="Photo has not been captured."), 404

    try:
        result = run_ai_pipeline(raw_path, base_name)
    except InferenceStageError as error:
        return jsonify(
            status="failed",
            stage=error.stage,
            error=str(error),
            previewImage=error.preview_image,
            canRecord=False,
        ), 422
    except Exception as error:
        return jsonify(error=str(error)), 500

    return jsonify(result)


@app.post("/api/audio")
def save_audio():
    base_name = request.form.get("baseName", "")
    audio = request.files.get("audio")

    if not BASE_NAME_RE.fullmatch(base_name):
        return jsonify(error="Invalid base name"), 400
    if audio is None:
        return jsonify(error="Missing audio file"), 400

    record_image_path = pending_record_path(base_name)
    record_kind_path = pending_record_kind_path(base_name)
    if not record_image_path.exists() or not record_kind_path.exists():
        return jsonify(error="Inference must finish before recording is saved."), 409

    audio_bytes = audio.read()
    if not audio_bytes:
        return jsonify(error="Empty audio file"), 400
    if len(audio_bytes) > 100 * 1024 * 1024:
        return jsonify(error="Audio is too large"), 413

    photo_filename = f"{base_name}.jpg"
    audio_filename = f"{base_name}.mp4"
    record_kind = record_kind_path.read_text(encoding="utf-8").strip()

    if record_kind == "keypoint":
        keypoint_image = PHOTO_KEYPOINT_DIR / photo_filename
        keypoint_audio = PHOTO_KEYPOINT_DIR / audio_filename
        object_image = PHOTO_DETECTION_DIR / photo_filename
        object_audio = PHOTO_DETECTION_DIR / audio_filename

        shutil.copy2(record_image_path, keypoint_image)
        convert_audio_to_mp4(audio_bytes, keypoint_audio)
        shutil.copy2(pending_detection_photo_path(base_name), object_image)
        shutil.copy2(keypoint_audio, object_audio)
        image_path = keypoint_image
        audio_path = keypoint_audio
    elif record_kind == "detection":
        image_path = PHOTO_DETECTION_DIR / photo_filename
        audio_path = PHOTO_DETECTION_DIR / audio_filename
        shutil.copy2(record_image_path, image_path)
        convert_audio_to_mp4(audio_bytes, audio_path)
    else:
        return jsonify(error="Unknown inference result type."), 409

    return jsonify(
        photoFilename=photo_filename,
        audioFilename=audio_filename,
        imagePath=str(image_path),
        audioPath=str(audio_path),
    )


@app.get("/api/health")
def health():
    return jsonify(status="ok")


def run_ai_pipeline(raw_path: Path, base_name: str) -> dict[str, object]:
    case_dir = RESULT_DIR / base_name
    case_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw_path, case_dir / "captured_original.jpg")

    raw_image_b64 = base64.b64encode(raw_path.read_bytes()).decode("ascii")
    original_preview = f"data:image/jpeg;base64,{raw_image_b64}"

    try:
        detection = post_image(
            AI_DETECT_CROP_URL,
            raw_path,
            f"{base_name}.jpg",
            {"crop_scale": CROP_SCALE},
        )
    except Exception as error:
        raise InferenceStageError(
            "detection",
            "物件偵測失敗，請重新拍攝",
            original_preview,
        ) from error

    detection_image_b64 = str(
        detection.get("detection_image_base64")
        or detection.get("annotated_image_base64")
        or ""
    )
    cropped_image_b64 = str(detection.get("cropped_image_base64") or "")
    if not detection_image_b64 or not cropped_image_b64:
        raise InferenceStageError(
            "detection",
            "物件偵測失敗，請重新拍攝",
            original_preview,
        )

    write_b64_image(pending_detection_photo_path(base_name), detection_image_b64)
    write_b64_image(case_dir / "object_detection.jpg", detection_image_b64)

    crop_path = PENDING_DIR / f"{base_name}_bbox_crop.jpg"
    write_b64_image(crop_path, cropped_image_b64)
    write_b64_image(case_dir / "bbox_crop_1_5x.jpg", cropped_image_b64)

    try:
        keypoint = post_image(
            AI_KEYPOINT_FOCUS_URL,
            crop_path,
            crop_path.name,
            {"focus_scale": KEYPOINT_FOCUS_SCALE},
        )
    except Exception as error:
        return save_partial_detection_result(
            base_name,
            case_dir,
            cropped_image_b64,
            detection,
            str(error),
        )

    focused_image_b64 = str(keypoint.get("focused_image_base64") or "")
    keypoint_image_b64 = str(
        keypoint.get("keypoint_image_base64")
        or keypoint.get("annotated_image_base64")
        or ""
    )
    if not focused_image_b64:
        return save_partial_detection_result(
            base_name,
            case_dir,
            cropped_image_b64,
            detection,
            "No focused keypoint image was returned.",
        )

    write_record_result(base_name, "keypoint", focused_image_b64)
    write_b64_image(case_dir / "final_cutting_edge_focus.jpg", focused_image_b64)
    if keypoint_image_b64:
        write_b64_image(case_dir / "keypoint_analysis_on_crop.jpg", keypoint_image_b64)

    result_payload: dict[str, object] = {
        "status": "success",
        "stage": "keypoint",
        "baseName": base_name,
        "previewImage": f"data:image/jpeg;base64,{focused_image_b64}",
        "canRecord": True,
        "photosWillSaveAs": {
            "original": str(PHOTO_ORIGINAL_DIR / f"{base_name}.jpg"),
            "objectDetectionImage": str(PHOTO_DETECTION_DIR / f"{base_name}.jpg"),
            "objectDetectionAudio": str(PHOTO_DETECTION_DIR / f"{base_name}.mp4"),
            "keypointImage": str(PHOTO_KEYPOINT_DIR / f"{base_name}.jpg"),
            "keypointAudio": str(PHOTO_KEYPOINT_DIR / f"{base_name}.mp4"),
        },
        "detection": detection,
        "keypoint": keypoint,
    }
    write_pipeline_json(case_dir, result_payload)
    return result_payload


def save_partial_detection_result(
    base_name: str,
    case_dir: Path,
    cropped_image_b64: str,
    detection: dict[str, object],
    detail: str,
) -> dict[str, object]:
    write_record_result(base_name, "detection", cropped_image_b64)
    result_payload: dict[str, object] = {
        "status": "partial",
        "stage": "keypoint",
        "baseName": base_name,
        "error": "關鍵點偵測失敗，可先錄音保存物件偵測結果",
        "detail": detail,
        "previewImage": f"data:image/jpeg;base64,{cropped_image_b64}",
        "canRecord": True,
        "photosWillSaveAs": {
            "original": str(PHOTO_ORIGINAL_DIR / f"{base_name}.jpg"),
            "objectDetectionImage": str(PHOTO_DETECTION_DIR / f"{base_name}.jpg"),
            "objectDetectionAudio": str(PHOTO_DETECTION_DIR / f"{base_name}.mp4"),
        },
        "detection": detection,
        "keypoint": None,
    }
    write_pipeline_json(case_dir, result_payload)
    return result_payload


def write_pipeline_json(case_dir: Path, payload: dict[str, object]) -> None:
    (case_dir / "pipeline_result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def decode_jpeg_data_url(data_url: str) -> bytes:
    if not data_url.startswith("data:image/jpeg;base64,"):
        raise ValueError("Invalid JPEG image")
    try:
        image_bytes = base64.b64decode(data_url.split(",", 1)[1], validate=True)
    except (ValueError, TypeError) as error:
        raise ValueError("Invalid image data") from error
    if len(image_bytes) > 25 * 1024 * 1024:
        raise ValueError("Image is too large")
    return image_bytes


def post_image(url: str, image_path: Path, filename: str, fields: dict[str, str]) -> dict[str, object]:
    boundary = f"----gechic-{uuid4().hex}"
    file_bytes = image_path.read_bytes()
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(form_field(boundary, name, value))
    parts.extend(
        [
            f"--{boundary}\r\n".encode("utf-8"),
            f'Content-Disposition: form-data; name="file"; filename="{Path(filename).name}"\r\n'.encode("utf-8"),
            b"Content-Type: image/jpeg\r\n\r\n",
            file_bytes,
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
    )
    return post_multipart(url, boundary, b"".join(parts))


def post_multipart(url: str, boundary: str, body: bytes) -> dict[str, object]:
    http_request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(http_request, timeout=240) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{url} returned {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"{url} is not reachable: {error.reason}") from error


def form_field(boundary: str, name: str, value: str) -> bytes:
    return (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
        f"{value}\r\n"
    ).encode("utf-8")


def write_b64_image(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(value, validate=True))


def convert_audio_to_mp4(audio_bytes: bytes, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir) / "recording_input"
        input_path.write_bytes(audio_bytes)
        command = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-vn",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as error:
            message = getattr(error, "stderr", "") or str(error)
            raise RuntimeError(f"Audio conversion failed: {message}") from error


def pending_raw_path(base_name: str) -> Path:
    return PENDING_DIR / f"{base_name}_captured.jpg"


def pending_record_path(base_name: str) -> Path:
    return PENDING_DIR / f"{base_name}_record.jpg"


def pending_record_kind_path(base_name: str) -> Path:
    return PENDING_DIR / f"{base_name}_record_kind.txt"


def pending_detection_photo_path(base_name: str) -> Path:
    return PENDING_DIR / f"{base_name}_object_detection.jpg"


def write_record_result(base_name: str, kind: str, image_b64: str) -> None:
    write_b64_image(pending_record_path(base_name), image_b64)
    pending_record_kind_path(base_name).write_text(kind, encoding="utf-8")


def ensure_https_certificate() -> tuple[str, str]:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    host_ip = os.environ.get("CAMERA_HOST_IP", "127.0.0.1").strip()
    try:
        ip_value = ipaddress.ip_address(host_ip)
    except ValueError:
        ip_value = ipaddress.ip_address("127.0.0.1")
        host_ip = "127.0.0.1"

    cert_path = CERT_DIR / "camera.crt"
    key_path = CERT_DIR / "camera.key"
    marker_path = CERT_DIR / "host_ip.txt"
    marker_value = f"v2:{host_ip}"
    if (
        cert_path.exists()
        and key_path.exists()
        and marker_path.exists()
        and marker_path.read_text(encoding="ascii").strip() == marker_value
    ):
        return str(cert_path), str(key_path)

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, f"ToolKnife Camera {host_ip}")]
    )
    now = datetime.now()
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=3650))
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.DNSName("localhost"),
                    x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                    x509.IPAddress(ip_value),
                ]
            ),
            critical=False,
        )
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=True,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=None,
                decipher_only=None,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    marker_path.write_text(marker_value, encoding="ascii")
    return str(cert_path), str(key_path)


if __name__ == "__main__":
    certificate = ensure_https_certificate()
    app.run(host="0.0.0.0", port=8000, threaded=True, ssl_context=certificate)
