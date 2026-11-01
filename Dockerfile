FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY gateway ./gateway
COPY eval_results.json .
RUN useradd --system --no-create-home app && mkdir /data && chown app /data
USER app
ENV PII_AUDIT_DB=/data/audit.db
EXPOSE 8000
# ONE worker on purpose: the token vault lives in memory (see README/DEPLOY.md).
CMD ["python", "-m", "uvicorn", "gateway.server:build_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
