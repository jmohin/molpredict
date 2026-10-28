# One container serving both halves of the app: the built frontend as static files and the
# FastAPI backend under /api, so the browser talks to a single origin and needs no CORS or proxy.

# --- Build the frontend -------------------------------------------------------------------
FROM node:22-slim AS frontend

WORKDIR /build
# Dependencies first, so editing a component doesn't re-run the install.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build


# --- Build the backend environment and assemble the image ---------------------------------
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# Same split again: the dependency layer is ~1 GB (torch, chemprop, rdkit) and only changes
# when the lockfile does. --no-dev leaves out pytest and httpx.
# The cache mount keeps uv's unpacked-wheel cache (~1.4 GB) out of the image layer while still
# making a rebuild reuse the downloads.
COPY backend/pyproject.toml backend/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --no-install-project
COPY backend/src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev

COPY --from=frontend /build/dist ./static

ENV PATH=/app/.venv/bin:$PATH \
    MOLPREDICT_STATIC=/app/static

# Nothing here writes to disk, so the app has no reason to run as root — and no reason to own
# /app either: chown -R would copy the whole 1.6 GB environment into a second image layer.
RUN useradd --create-home --uid 1000 app
USER app

EXPOSE 8000

# Loading the models takes ~8s, so the first check is given room before the container is
# called unhealthy.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"

CMD ["uvicorn", "molpredict_backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
