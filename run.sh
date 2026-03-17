docker run -d -p 80:8080 --memory="600m" --restart on-failure \
  -v "$HOME/.config/gcloud/application_default_credentials.json:/app/adc.json:ro" \
  -e TIMEOUT=600 \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/adc.json \
  -e GOOGLE_CLOUD_PROJECT=project-b0e0c2c6-984d-4ad9-b61  \
  -e GCV_BUCKET=ocr-storage-gcv \
  -e GCV_PREFIX=uploads \
  -e SECRET_API_KEY=$SECRET_API_KEY \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  ghcr.io/darkstar1997/ocr-engine
