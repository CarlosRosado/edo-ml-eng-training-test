FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app  

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the src directory
COPY ../src /app/src

# Copy the pipeline_mlflow.py script from src/pipeline
COPY ../src/pipeline/pipeline_mlflow.py /app/src/pipeline/pipeline_mlflow.py

# Default command to run the training pipeline
CMD ["python", "/app/src/pipeline/pipeline_mlflow.py"]