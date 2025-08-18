# Base Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirement files and install dependencies
COPY app/requirements.txt app/requirements.txt
COPY training/requirements.txt training/requirements.txt

RUN pip install --upgrade pip \
    && pip install -r app/requirements.txt \
    && pip install -r training/requirements.txt \
    && pip install pytest

# Copy all code into container
COPY app/ app/
COPY training/ training/
COPY evaluation/ evaluation/
COPY pipelines/ pipelines/
COPY tests/ tests/

# Default command (can be overridden by Cloud Build)
CMD ["bash"]
