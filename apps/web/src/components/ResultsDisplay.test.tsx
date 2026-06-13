import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ResultsDisplay } from '@/components/ResultsDisplay';
import { getDefaultParams } from '@/lib/storage';
import type { BacktestResponse } from '@/lib/types';

// The Plotly chart pulls in a large client-only bundle; stub it for the test.
vi.mock('@/components/PlotlyChart', () => ({
  PlotlyChart: () => <div data-testid="chart" />,
}));

const results: BacktestResponse = {
  metrics: {
    cagr: 0.12,
    sharpe: 1.35,
    max_drawdown: -0.2,
    win_rate: 0.55,
    total_trades: 8,
    avg_trade_return: 0.03,
    volatility: 0.18,
  },
  equity_curve: [
    { date: '2020-01-01', equity: 100 },
    { date: '2020-01-02', equity: 110 },
  ],
  benchmark_curve: [
    { date: '2020-01-01', equity: 100 },
    { date: '2020-01-02', equity: 105 },
  ],
  trades: [{ date: '2020-01-02', side: 'long', price: 101.5, quantity: 10 }],
  params_used: { ...getDefaultParams(), ticker: 'SPY', strategy: 'ema_crossover' },
};

describe('ResultsDisplay', () => {
  it('renders the metric board with formatted values', () => {
    render(<ResultsDisplay results={results} />);
    expect(screen.getByText('CAGR')).toBeInTheDocument();
    expect(screen.getByText('12.00%')).toBeInTheDocument();
    expect(screen.getByText('1.35')).toBeInTheDocument();
    expect(screen.getByText('EMA Crossover')).toBeInTheDocument();
  });

  it('renders trade blotter rows', () => {
    render(<ResultsDisplay results={results} />);
    expect(screen.getByText('Long')).toBeInTheDocument();
    expect(screen.getByText(/101\.50/)).toBeInTheDocument();
  });
});
