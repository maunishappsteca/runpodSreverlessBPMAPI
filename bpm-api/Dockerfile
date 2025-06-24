# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required by librosa & soundfile
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy Python files
COPY requirements.txt .
COPY app.py .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Flask runs on
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]
