FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

# Install dependencies first so this layer is cached until requirements change.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY data ./data

# Train at build time so the model always matches the installed scikit-learn version.
RUN python -m app.training

# Run as a non-root user.
RUN useradd --create-home appuser && chown -R appuser /srv
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
