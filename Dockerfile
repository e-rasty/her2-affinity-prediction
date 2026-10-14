FROM python:3.12-slim

WORKDIR /app

ENV PROJECT_ROOT=/app

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir .

COPY artifacts/ artifacts/

CMD ["uvicorn", "her2clf.api:app", "--host", "0.0.0.0", "--port", "8000"]