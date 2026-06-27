# Image for the Netflix Retention DSS (pipeline + Streamlit dashboard).
FROM python:3.11-slim

# libgomp1 is required at runtime by XGBoost.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (better layer caching).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source (bind mount in compose overrides this for live edits).
COPY . .

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

EXPOSE 8501

# Default: serve the dashboard. The pipeline service overrides this command.
CMD ["streamlit", "run", "app/Home.py", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
