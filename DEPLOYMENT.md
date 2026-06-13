# Deployment

The app ships as a single container: the Next.js frontend is built to a static
export and served by the FastAPI backend from the same origin. One image, one
URL, no separate frontend host and no CORS to configure.

## How the single container works

1. A Node build stage runs `next build` with `output: 'export'`, producing a
   static site in `apps/web/out`.
2. The Python runtime stage installs the API dependencies, copies the API code,
   and copies the static export into `/app/static`.
3. At startup FastAPI mounts `/app/static` at `/`. API routes (`/backtest`,
   `/api/*`, `/health`, `/docs`) are registered first and always take
   precedence; every other path serves the frontend.

The browser only ever talks to one origin, so `runBacktest` calls the API with a
relative path in production. No `NEXT_PUBLIC_API_URL` is required for the
combined deployment.

## Build and run locally with Docker

```bash
# From the repository root
docker build -t quant-terminal .
docker run --rm -p 8080:8080 quant-terminal
# Open http://localhost:8080
```

The container honors the `PORT` environment variable (defaults to 8080), which
is what most managed platforms inject.

## Environment variables

All optional. The defaults work for a public, single-container deployment.

| Variable          | Default            | Purpose                                                        |
| ----------------- | ------------------ | -------------------------------------------------------------- |
| `PORT`            | `8080`             | Port the server binds to (set automatically by most hosts).   |
| `ALLOWED_ORIGINS` | `*`                | Comma separated CORS origins. Same-origin, so `*` is fine.    |
| `API_KEY`         | unset              | If set, non-public endpoints require a matching `X-API-Key`.  |
| `DATA_CACHE_DIR`  | system temp dir    | On-disk market-data cache location.                           |
| `DATA_CACHE_TTL`  | `3600`             | Cache freshness in seconds (`0` disables caching).            |

## Hosting options

Any platform that builds a Dockerfile will work. Pick one.

### Render (blueprint included)

A `render.yaml` blueprint is committed at the repository root.

1. Push this repository to GitHub.
2. In the Render dashboard choose New > Blueprint and select the repo.
3. Render reads `render.yaml`, builds the Dockerfile, and deploys. It injects
   `PORT` and uses `/health` as the health check.
4. The live URL appears when the first deploy finishes.

To require an API key, set `API_KEY` on the service (the blueprint leaves it
unset and unsynced) and provide the same value to the frontend at build time via
`NEXT_PUBLIC_API_KEY`.

### Railway

1. New Project > Deploy from GitHub repo.
2. Railway detects the Dockerfile and builds it. `PORT` is provided
   automatically.
3. Generate a domain under the service Settings > Networking.

### Fly.io

```bash
fly launch --no-deploy   # generates fly.toml; keep the internal port at 8080
fly deploy
```

Fly sets `PORT` to the internal port (8080), which the container already uses.

### Google Cloud Run

```bash
gcloud run deploy quant-terminal \
  --source . \
  --region us-west1 \
  --allow-unauthenticated \
  --port 8080
```

Cloud Run builds the Dockerfile, injects `PORT`, and returns the service URL.

## Local development (two processes)

For day-to-day development you do not need Docker. Run the API and the Next.js
dev server side by side so you get hot reload on both:

```bash
npm install
npm run dev
# Web: http://localhost:3000   API: http://localhost:8080
```

In development the frontend targets `http://localhost:8080` automatically. Set
`NEXT_PUBLIC_API_URL` only if your API runs somewhere else.

## Testing

```bash
# API tests (coverage enforced by pytest.ini)
cd apps/api
pip install -r requirements-dev.txt
python -m pytest

# Frontend lint / typecheck / test / build (from repo root)
npx turbo run lint typecheck test build
```

## Project structure

```
ema-sharpe-dashboard/
  apps/
    web/                  # Next.js frontend (static export)
      src/
        app/              # App router
        components/       # React components
        lib/              # Utilities and API client
      next.config.js      # output: 'export'
      package.json
    api/                  # FastAPI backend (also serves the built frontend)
      main.py             # Application, endpoints, static mount
      tests/              # Unit tests
      requirements.txt
      Dockerfile          # API-only image (optional)
  packages/
    types/                # Shared TypeScript types
  scripts/                # Local dev helpers
  .github/workflows/      # CI
  Dockerfile              # Single-container build (frontend + API)
  render.yaml             # Render blueprint
  package.json            # Root workspace config
  turbo.json              # Turbo configuration
```

## API endpoints

- `GET /health` - Health check
- `GET /api/strategies` - List strategies and parameters
- `POST /backtest`, `POST /api/backtest` - Run a backtest
- `POST /api/monte-carlo` - Bootstrap Monte Carlo of terminal equity
- `POST /api/rolling` - Rolling Sharpe / return / drawdown
- `POST /api/benchmarks` - Strategy vs. buy-and-hold summary
- `POST /api/parameter-sweep` - Strategy-aware Sharpe heatmap
- `POST /api/export.csv` - Equity curve as CSV

## License

MIT
