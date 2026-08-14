# Use a slim Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed by TensorFlow / image processing / awscli (if used)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    unzip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (leverages Docker layer caching)
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose the Flask port
EXPOSE 8080

# Environment variables (override at runtime with docker run -e or docker-compose)
ENV FLASK_APP=app.py
ENV PYTHONUNBUFFERED=1

# Run the app
CMD ["python", "app.py"]