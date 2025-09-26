FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY src/ ./src/
COPY pyproject.toml .

# Install the package in development mode
RUN pip install -e .

# Copy the app entry point and landing page
COPY app.py .
COPY index.html .

# Set environment variables
ENV PYTHONPATH=/app/src
ENV WANDB_SILENT=True
ENV WEAVE_SILENT=True
ENV MCP_TRANSPORT=http
ENV HOST=0.0.0.0

# Set W&B cache directories to writable locations
ENV WANDB_CACHE_DIR=/tmp/.wandb_cache
ENV WANDB_CONFIG_DIR=/tmp/.wandb_config
ENV WANDB_DATA_DIR=/tmp/.wandb_data
ENV HOME=/tmp

# Expose port for HTTP transport
EXPOSE 7860

# Run with single worker using Uvicorn's async event loop
# MCP protocol requires stateful session management (in-memory sessions)
# Single async worker handles high concurrency via event loop (1000+ concurrent connections)
CMD ["uvicorn", "app:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--log-level", "info", \
     "--timeout-keep-alive", "120", \
     "--limit-concurrency", "1000"]
