# Use Python 3.12 slim base
FROM python:3.12-slim

# Basic env hygiene
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends dos2unix

EXPOSE 8080

RUN pip install --no-cache-dir \
    fastapi==0.118.0 uvicorn==0.37.0 gunicorn==23.0.0 openai==2.1.0 pymupdf==1.26.4 pillow python-docx python-multipart==0.0.20 google-cloud-vision==3.10.2 google-cloud-storage==3.4.0

WORKDIR /app

COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh
RUN dos2unix /app/start.sh

COPY app.py /app/app.py

CMD ["/app/start.sh"]
