FROM python:3.12-slim

# Install ffmpeg (needed for --video audio extraction)
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install only the dependencies propose_cuts actually needs
RUN uv pip install --system \
    deepgram-sdk \
    langchain-openai \
    pydantic \
    httpx

# Copy the edit package so we can run propose_cuts as a module
COPY src/ ./src/
ENV PYTHONPATH=/app

ENTRYPOINT ["python", "-m", "src.edit.propose_cuts"]
