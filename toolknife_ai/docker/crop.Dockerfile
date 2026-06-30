FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/app/data/ultralytics_config

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

FROM base AS runtime

ENV TOOLKNIFE_CROP_HOST=0.0.0.0 \
    TOOLKNIFE_CROP_PORT=8004

EXPOSE 8004

CMD ["python", "-m", "uvicorn", "crop_api.crop_server:app", "--host", "0.0.0.0", "--port", "8004"]
