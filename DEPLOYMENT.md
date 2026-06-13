# EMA + Sharpe Dashboard

## Development Setup

### Prerequisites
- Node.js 18+
- Python 3.11+
- npm or yarn

### Local Development

1. **Clone and install dependencies:**
   ```bash
   git clone <your-repo-url>
   cd ema-sharpe-dashboard
   # Backend deps
   pip install -r apps/api/requirements.txt
   # Frontend deps (optional; Next.js)
   npm install
   ```

2. **Start the development servers:**
   ```bash
   # Backend (FastAPI)
   cd apps/api && python -m uvicorn main:app --reload --port 8080
   
   # Frontend (Next.js, optional)
   cd apps/web && npm run dev
   ```

3. **Access the application:**
- Backend API: http://localhost:8080
- API Docs: http://localhost:8080/docs
- Frontend (if running): http://localhost:3000

### Environment Variables

Create `apps/web/.env.local` (only if using the Next.js frontend):
```bash
NEXT_PUBLIC_API_URL=http://localhost:8080
```

## Production Deployment

### Backend (Cloud Run)

1. Build and push image:
   ```bash
   gcloud builds submit --tag gcr.io/$PROJECT_ID/ema-sharpe
   ```
2. Deploy:
   ```bash
   gcloud run deploy ema-sharpe \
     --image gcr.io/$PROJECT_ID/ema-sharpe \
     --platform managed \
     --region us-west1 \
     --allow-unauthenticated \
     --port 8080
   ```

### Frontend (Vercel or Cloud Run)

1. Set environment variable:
   - `NEXT_PUBLIC_API_URL=https://<your-api-url>`
2. Deploy the Next.js app (Vercel recommended) or build a separate container from `apps/web/Dockerfile`.

## Testing

```bash
# API tests (with coverage; enforced by pytest.ini)
cd apps/api
pip install -r requirements-dev.txt
python -m pytest

# Frontend lint / typecheck / build (from repo root)
npx turbo run lint typecheck build
```

## Project Structure

```
ema-sharpe-dashboard/
  apps/
    web/                  # Next.js frontend
      src/
        app/              # App router
        components/       # React components
        lib/             # Utilities and API client
      package.json
      Dockerfile
    api/                  # FastAPI backend
      main.py             # Application and endpoints
      tests/              # Unit tests
      requirements.txt
      Dockerfile
  packages/
    types/                # Shared TypeScript types
  scripts/                # Local dev helpers
  .github/workflows/      # CI/CD
  package.json            # Root workspace config
  turbo.json              # Turbo configuration
  README.md
```

## Key Features

- **Five Strategies**: EMA Crossover, RSI Mean Reversion, SMA Crossover,
  Bollinger Breakout, Momentum
- **Realistic Costs**: Fees and slippage simulation
- **Volatility Targeting**: Optional risk management
- **Performance Metrics**: CAGR, Sharpe, Max DD, Win Rate, volatility
- **Benchmark**: Strategy vs. buy-and-hold
- **Interactive Charts**: Plotly equity curve with benchmark + trade markers
- **Parameter Persistence**: Saves user preferences (local storage)
- **Rate Limiting**: 60 req/min per IP
- **Error Handling**: Graceful error messages with proper status codes

## API Endpoints

- `GET /health` - Health check
- `GET /api/strategies` - List strategies and parameters
- `POST /backtest`, `POST /api/backtest` - Run a backtest
- `POST /api/monte-carlo` - Bootstrap Monte Carlo of terminal equity
- `POST /api/rolling` - Rolling Sharpe / CAGR / drawdown
- `POST /api/benchmarks` - Strategy vs. buy-and-hold summary
- `POST /api/parameter-sweep` - Grid sweep over fast/slow parameters
- `POST /api/export.csv` - Equity curve as CSV

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT
