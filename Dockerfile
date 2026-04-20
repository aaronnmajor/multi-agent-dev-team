FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run as a non-root user for defence-in-depth.
RUN useradd --create-home --uid 1000 agent \
    && chown -R agent:agent /app
USER agent

# Healthcheck: imports the graph module to verify dependencies resolve.
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "from orchestration.graph import create_graph; create_graph()" || exit 1

CMD ["python", "-m", "orchestration.graph"]
