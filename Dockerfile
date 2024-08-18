# Use an official CUDA runtime as a parent image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Set the working directory in the container
WORKDIR /app

# Install Python and other necessary packages
RUN apt-get update && apt-get install -y python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Add a non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy only the requirements file to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

# Copy the rest of the application code
COPY app .

# Change ownership of the app directory to the non-root user
RUN chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser

# Set environment variables
ENV BIG_MODEL=false
ENV CUDA=true

# Define the command to run the application
CMD ["uvicorn", "--host", "0.0.0.0", "main:app"]
