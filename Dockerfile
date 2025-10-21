# Dockerfile for Parkinson's Disease Detection System
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (including build tools for praat-parselmouth)
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
RUN mkdir -p logs models data/raw/speech data/raw/handwriting data/raw/gait data/processed

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV FLASK_ENV=production

# Generate synthetic handwriting and gait datasets
RUN python3 generate_modality_datasets.py

# Train all models (speech data already in repo from git)
RUN python3 train.py && \
    python3 train_handwriting_model.py && \
    python3 train_gait_model.py && \
    cp models/best_model.joblib models/speech_model.joblib && \
    cp models/scaler.joblib models/speech_scaler.joblib

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-", "wsgi:app"]

