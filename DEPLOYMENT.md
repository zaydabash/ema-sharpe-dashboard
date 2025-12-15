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
# API tests
cd apps/api
python -m pytest tests/ -v

# Frontend tests (optional, requires Next.js deps)
cd apps/web
npm test

# Type checking (frontend)
npm run typecheck
```

## Project Structure

```
ema-sharpe-dashboard/
├── apps/
│   ├── web/                 # Next.js frontend
│   │   ├── src/
│   │   │   ├── app/         # App router
│   │   │   ├── components/  # React components
│   │   │   └── lib/        # Utilities
│   │   ├── package.json
│   │   └── Dockerfile
│   └── api/                 # FastAPI backend
│       ├── main.py          # Main application
│       ├── tests/           # Unit tests
│       ├── requirements.txt
│       └── Dockerfile
├── packages/
│   └── types/               # Shared TypeScript types
├── .github/workflows/       # CI/CD
├── package.json            # Root package.json
├── turbo.json              # Turbo configuration
└── README.md
```

## Key Features

- **EMA Crossover Strategy**: Fast/slow EMA signals
- **Realistic Costs**: Fees and slippage simulation
- **Volatility Targeting**: Optional risk management
- **Performance Metrics**: CAGR, Sharpe, Max DD, Win Rate
- **Interactive Charts**: Plotly visualization
- **Responsive Design**: Mobile-friendly UI
- **Parameter Persistence**: Saves user preferences
- **Rate Limiting**: API protection
- **Error Handling**: Graceful error messages

## API Endpoints

- `GET /health` - Health check
- `POST /backtest` - Run backtest

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT
