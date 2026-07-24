# 1. Build the React JS Frontend
FROM node:20-alpine AS frontend-builder
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# 2. Build the Python REST API backend
FROM python:3.11-slim-bookworm

# Prevent Python from writing pyc files to disk and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Pre-create the directory for volume persistence
RUN mkdir -p /app/data

# Install minimal system utilities required for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend Python codebase
COPY src/ ./src/

# Copy the built React assets into the FastAPI static files directory
COPY --from=frontend-builder /frontend/dist/ ./src/static/

# API default port exposure
EXPOSE 8000

# Command to execute the application natively inside the container
ENTRYPOINT ["python3", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]