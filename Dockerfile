FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Default entrypoint for Cloud Build test step
ENTRYPOINT ["python"]
