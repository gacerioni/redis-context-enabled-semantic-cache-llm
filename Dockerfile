FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only the minimal requirements file
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy the app source
COPY . .

EXPOSE 7860
ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "src.app"]