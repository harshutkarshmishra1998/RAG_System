# ---------------------------
# Base image
# ---------------------------
FROM python:3.11-slim

# ---------------------------
# System dependencies
# ---------------------------
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    ghostscript \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------
# Environment variables
# ---------------------------
# Poppler binaries live in /usr/bin
# Tesseract binary is /usr/bin/tesseract
ENV POPPLER_PATH=/usr/bin
ENV TESSERACT_PATH=/usr/bin/tesseract

# ---------------------------
# Install uv (fast & reproducible)
# ---------------------------
RUN pip install --no-cache-dir --upgrade pip uv

# ---------------------------
# Python dependencies (from pyproject.toml)
# ---------------------------
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# ---------------------------
# Application code
# ---------------------------
COPY . .

# ---------------------------
# Default command
# ---------------------------
CMD ["python", "-m", "main"]