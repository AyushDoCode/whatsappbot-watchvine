# Multi-service Docker image for WhatsApp Bot with Product Search
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    wget \
    curl \
    git \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install numpy first to ensure correct version for FAISS
RUN pip install --no-cache-dir "numpy<2"

# Install additional dependencies for scheduler and image processing
RUN pip install --no-cache-dir \
    schedule==1.2.0 \
    uvicorn==0.27.0 \
    fastapi==0.109.0 \
    faiss-cpu==1.7.4 \
    sentence-transformers==2.3.1 \
    Pillow==10.2.0 \
    beautifulsoup4==4.12.3

# Copy application code
COPY . .

# Ensure image_identifier folder and its contents are present
COPY image_identifier/ /app/image_identifier/

# Create necessary directories
RUN mkdir -p /app/temp_images /app/logs

# Copy and set permissions for startup script
COPY startup.sh /app/startup.sh
RUN chmod +x /app/startup.sh && \
    sed -i 's/\r$//' /app/startup.sh

# Create supervisord configuration
RUN echo '[supervisord]\n\
nodaemon=true\n\
logfile=/app/logs/supervisord.log\n\
pidfile=/var/run/supervisord.pid\n\
\n\
[program:main]\n\
command=python main.py\n\
directory=/app\n\
autostart=true\n\
autorestart=true\n\
stderr_logfile=/app/logs/main.err.log\n\
stdout_logfile=/app/logs/main.out.log\n\
environment=PYTHONUNBUFFERED=1\n\
startsecs=10\n\
\n\
[program:text_search_api]\n\
command=python text_search_api.py\n\
directory=/app\n\
autostart=true\n\
autorestart=true\n\
stderr_logfile=/app/logs/text_search_api.err.log\n\
stdout_logfile=/app/logs/text_search_api.out.log\n\
environment=PYTHONUNBUFFERED=1\n\
startsecs=10\n\
\n\
[program:image_identifier_api]\n\
command=python api.py\n\
directory=/app/image_identifier\n\
autostart=true\n\
autorestart=true\n\
stderr_logfile=/app/logs/image_identifier_api.err.log\n\
stdout_logfile=/app/logs/image_identifier_api.out.log\n\
environment=PYTHONUNBUFFERED=1\n\
startsecs=30\n\
\n\
[program:nightly_scraper]\n\
command=python nightly_scraper_scheduler.py\n\
directory=/app\n\
autostart=true\n\
autorestart=true\n\
stderr_logfile=/app/logs/nightly_scraper.err.log\n\
stdout_logfile=/app/logs/nightly_scraper.out.log\n\
environment=PYTHONUNBUFFERED=1\n\
startsecs=10' > /etc/supervisor/conf.d/supervisord.conf

# Expose ports
# 5000 - Main Flask app
# 8001 - Text Search API
# 8002 - Image Identifier API
# 8080 - Reserved for Evolution API (external)
EXPOSE 5000 8001 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run startup script
CMD ["/app/startup.sh"]
