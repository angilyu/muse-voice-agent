# Retell-backed MCP server image (used by Render; works on any container host).
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /bin/uv

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=10000

# Install the locked runtime deps (no dev group, no livekit extra) from public PyPI.
ARG PYPI_INDEX_URL=https://pypi.org/simple
COPY pyproject.toml uv.lock ./
RUN uv export --frozen --no-dev --no-emit-project -o /tmp/requirements.txt \
    && uv pip install --system --no-cache --index-url "$PYPI_INDEX_URL" -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

COPY src ./src

EXPOSE 10000
CMD ["python", "-m", "muse_voice_agent.mcp_server"]
