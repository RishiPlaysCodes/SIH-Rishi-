# Container image for Sentinel-X. Works on Hugging Face Spaces (Docker SDK),
# Fly.io, Google Cloud Run, Railway, and anywhere else that runs a container.
#
# The app is pure standard library, so there are NO dependencies to install —
# this image is tiny and builds in seconds.
FROM python:3.11-slim

WORKDIR /app
COPY . .

# The server honours $PORT (Cloud Run / Fly set it); default 7860 matches
# Hugging Face Spaces' expected port. It always binds 0.0.0.0.
ENV PORT=7860 \
    PYTHONUNBUFFERED=1
EXPOSE 7860

# Fail fast at build time if anything is syntactically broken.
RUN python -m compileall sentinelx

CMD ["python", "-m", "sentinelx.cli", "serve"]
