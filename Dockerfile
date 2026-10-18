FROM python:3.11-slim

WORKDIR /app

# Системные библиотеки для OpenCV (libgl1-mesa-glx в актуальном Debian больше нет)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CPU-сборка PyTorch (для GPU замените индекс на нужную версию CUDA)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY ml_pipeline/feature_extractor.py ml_pipeline/yolov8n.pt ./ml_pipeline/
COPY ml_pipeline/models/ ./ml_pipeline/models/

# Веса EfficientNet-B4 скачиваются один раз при сборке; дальше сервис работает без интернета
RUN python -c "import timm; timm.create_model('efficientnet_b4', pretrained=True, num_classes=0)"
ENV HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1

RUN mkdir -p /app/data/uploads /app/data/results /app/data/evidence_frames
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# API_KEY передаётся при запуске (docker compose читает его из .env)
CMD ["python", "backend/api.py"]
