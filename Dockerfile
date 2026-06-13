# Single-container build: compile the Next.js frontend to a static export, then
# serve it from the FastAPI backend so the whole app runs from one URL.

# --- Stage 1: build the static frontend --------------------------------------
FROM node:20-slim AS web
WORKDIR /repo

# Install workspace dependencies (cached unless manifests change).
COPY package.json package-lock.json turbo.json ./
COPY apps/web/package.json ./apps/web/package.json
COPY packages ./packages
RUN npm install

# Build the static export (outputs to apps/web/out).
COPY apps/web ./apps/web
RUN npm run build --workspace web

# --- Stage 2: API runtime that serves the frontend ---------------------------
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY apps/api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY apps/api/ /app
COPY --from=web /repo/apps/web/out /app/static

EXPOSE 8080

# Honor the platform-provided PORT (Cloud Run, Render, Railway, etc.).
CMD ["sh", "-c", "python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
