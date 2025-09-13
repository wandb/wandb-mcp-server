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

# Copy the app entry point
COPY app.py .

# Set environment variables
ENV PYTHONPATH=/app/src
ENV WANDB_SILENT=True
ENV WEAVE_SILENT=True

# Expose port for HTTP transport
EXPOSE 8080

# Run the application
CMD ["python", "app.py"]
