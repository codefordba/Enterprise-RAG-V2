# 1. Use an official lightweight Python runtime
FROM python:3.11-slim-bookworm

# 2. Prevent Python from writing pyc files to disk and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 3. Install minimal system utilities required for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy and install dependencies first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your internal source code into the container
COPY src/ ./src/

# 6. Streamlit default port exposure
EXPOSE 8501

# 7. Command to execute the application natively inside the container
ENTRYPOINT ["python3", "-m", "streamlit", "run", "src/main.py", "--server.port=8501", "--server.address=0.0.0.0"]