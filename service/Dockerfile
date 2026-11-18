# Build from the prepayment-and-avm-model-validation folder:
#   docker build -f service/Dockerfile -t valuation-service .
# Run with the model mounted read-only (the model file is git-ignored and never baked into the image):
#   docker run -p 8000:8000 -v "$PWD/data_work/p2/c1_gbm.txt:/models/c1_gbm.txt:ro" valuation-service
# NOTE: written but not built or run locally (Docker is not installed on the development machine).
FROM python:3.13-slim

# LightGBM needs the OpenMP runtime on Debian-based images
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY service/requirements.txt service/requirements.txt
RUN pip install --no-cache-dir -r service/requirements.txt
COPY service/ service/

ENV MODEL_PATH=/models/c1_gbm.txt \
    REFERENCE_PATH=/app/service/reference/psi_reference.json
RUN useradd --create-home appuser
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
