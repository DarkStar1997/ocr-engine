# Use Python 3.12 slim base
FROM python:3.12-slim

# Basic env hygiene
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends dos2unix

EXPOSE 8080

RUN pip install --no-cache-dir \
    fastapi uvicorn gunicorn openai pymupdf pillow python-docx python-multipart

WORKDIR /app

COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh
RUN dos2unix /app/start.sh

COPY app.py /app/app.py

CMD ["/app/start.sh"]
