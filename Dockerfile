# Parkinson's Disease Prediction System - Light mode only (custom logic, no ML libs)
FROM python:3.11-slim

WORKDIR /app

# Only curl for healthcheck; no ffmpeg/build-essential (not needed in light mode)
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Light requirements only (Flask, Waitress, auth, MongoDB)
COPY requirements-light.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-light.txt

COPY . .

RUN mkdir -p logs models data/raw/speech data/raw/handwriting data/raw/gait

ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV FLASK_ENV=production
ENV USE_LIGHT_MODE=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["python", "wsgi.py"]
