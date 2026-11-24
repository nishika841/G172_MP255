FROM python:3.12-slim

WORKDIR /app

# Install pinned dependencies first for efficient layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code, champion model, and configurations
COPY app/ ./app/
COPY models/ ./models/
COPY src/ ./src/
COPY params.yaml .
COPY pytest.ini .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
