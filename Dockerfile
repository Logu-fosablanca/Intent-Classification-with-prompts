FROM python:3.11-slim

WORKDIR /app

# System deps for sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY web/requirements.txt /app/web/requirements.txt
RUN pip install --no-cache-dir -r web/requirements.txt

COPY query_classifier/ /app/query_classifier/
COPY web/ /app/web/

# HF Spaces requires port 7860
EXPOSE 7860

CMD ["uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "7860"]
