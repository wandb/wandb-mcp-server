# Use Chainguard base image for enhanced security
FROM us-central1-docker.pkg.dev/wandb-mcp-production/chainguard-pull-through/coreweave/chainguard-base:latest

# Set working directory
WORKDIR /app

# Install Python and build dependencies
RUN apk add --no-cache \
    python3 \
    py3-pip \
    python3-dev \
    build-base \
    git \
    curl \
    ca-certificates

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY src/ ./src/
COPY pyproject.toml .

# Install the package in production mode (not editable)
RUN pip install .

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

# Create non-root user and set ownership
RUN adduser -D -u 1000 wandb && \
    chown -R wandb:wandb /app /tmp

# Switch to non-root user
USER wandb

# Expose port for HTTP transport (Cloud Run uses PORT env var)
EXPOSE 8080

# Run the Python app directly so it can read PORT env var
# MCP protocol requires stateful session management (in-memory sessions)
# Single async worker handles high concurrency via event loop (1000+ concurrent connections)
CMD ["python", "app.py"]
