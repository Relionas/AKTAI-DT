FROM python:3.10-slim

WORKDIR /app

# Установка системных зависимостей для OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Копирование requirements
COPY backend/requirements.txt .

# Установка Python зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода приложения
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Создание директорий для данных
RUN mkdir -p /app/data/uploads \
             /app/data/results \
             /app/data/evidence_frames \
             /app/logs

# Переменные окружения
ENV PYTHONUNBUFFERED=1
ENV API_KEY=fortebank-datathon-2026-team-alpha

# Экспорт порта
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Запуск API сервера
CMD ["python", "backend/api.py"]
