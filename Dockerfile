# DarGlobal × Wasalt Property Agent
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    EMBED_CACHE=/app/.cache/fastembed

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scraper ./scraper
COPY scripts ./scripts
COPY data/raw ./data/raw

# Build the knowledge base and the embedding index at image build time:
# raw scraped JSON -> processed datasets -> dense vectors (model is cached in the image).
RUN python scraper/build_kb.py && python scripts/build_index.py

# Run as non-root (uid 1000 is also what Hugging Face Spaces expects)
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"7860\")}/api/health')"
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
