FROM python:3.11-slim
WORKDIR /app
ENV PYTHONPATH=/app/src PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY tests ./tests
COPY pyproject.toml .
# Mount your downloaded dataset at /app/data and read results from /app/reports:
#   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/reports:/app/reports" retail-ml
CMD ["python", "-m", "retail_ml.pipeline", "--data-dir", "data", "--output-dir", "reports"]
