# Dockerfile for Parkinson's Disease Prediction System
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    gcc \
    g++ \
    make \
    libsndfile1 \
    ffmpeg \
    libavcodec-extra \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs models data/raw/speech data/raw/handwriting data/raw/gait

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV FLASK_ENV=production

# Generate multimodal datasets and train models
RUN python3 generate_modality_datasets.py && \
    python3 train.py && \
    cp models/best_model.joblib models/speech_model.joblib && \
    cp models/scaler.joblib models/speech_scaler.joblib && \
    python3 train_dl.py

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-", "wsgi:app"]

