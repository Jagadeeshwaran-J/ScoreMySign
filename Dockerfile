# Use full python image - already has gcc, build tools, no apt-get needed
FROM python:3.11

WORKDIR /app

# Set env vars to prevent .pyc files and enable log flushing
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Copy requirements first (layer cache optimization)
COPY requirements.txt .

# Install Python dependencies — no apt-get needed at all
# opencv-python-headless bundles its own libGL/libglib statically
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app/ ./app/

# Create non-root user
RUN useradd -m -u 1001 appuser && \
    mkdir -p /app/images /app/outputs && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]