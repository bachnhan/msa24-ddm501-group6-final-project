# =============================================================================
# Dockerfile for Customer Churn Prediction API with Monitoring
# DDM501 - Final Project
# =============================================================================

FROM python:3.10-slim

# Set working directory
# Memory Optimization for Render Free Tier (512MB)
ENV MALLOC_ARENA_MAX=2
ENV OMP_NUM_THREADS=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
  curl \
  build-essential \
  python3-dev \
  && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for cache optimization)
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Copy application code and models
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY models/ ./models/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
