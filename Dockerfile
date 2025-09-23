
FROM python:3.12-slim

WORKDIR /app

# System deps for pypdf and friends (very light)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install only the minimal runtime deps
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy the app
COPY . .

EXPOSE 7860
ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "src.app"]