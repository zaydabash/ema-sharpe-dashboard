'use client';

import { BacktestResponse } from '@/lib/types';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { PlotlyChart } from '@/components/PlotlyChart';

interface ResultsDisplayProps {
  results: BacktestResponse;
}

export function ResultsDisplay({ results }: ResultsDisplayProps) {
  const { metrics, equity_curve, trades } = results;

  const formatPercentage = (value: number) => `${(value * 100).toFixed(2)}%`;
  const formatNumber = (value: number) => value.toFixed(2);

  // Handle both snake_case and PascalCase field names
  const cagr = metrics.cagr ?? metrics.CAGR ?? 0;
  const sharpe = metrics.sharpe ?? metrics.Sharpe ?? 0;
  const maxDrawdown = metrics.max_drawdown ?? metrics.MaxDrawdown ?? 0;
  const winRate = metrics.win_rate ?? metrics.WinRate ?? 0;
  const totalTrades = metrics.total_trades ?? metrics.Trades ?? 0;

  return (
    <div className="space-y-6">
      {/* Metrics Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              CAGR
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {formatPercentage(cagr)}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Sharpe Ratio
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {formatNumber(sharpe)}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Max Drawdown
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-red-600">
              {formatPercentage(maxDrawdown)}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Win Rate
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {formatPercentage(winRate)}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Additional Metrics */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Total Trades
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-xl font-bold">
              {totalTrades}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Avg Trade Return
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-xl font-bold">
              {formatPercentage(metrics.avg_trade_return)}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Volatility
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-xl font-bold">
              {formatPercentage(metrics.volatility)}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Equity Curve Chart */}
      <Card>
        <CardHeader>
          <CardTitle>Equity Curve</CardTitle>
          <CardDescription>
            Portfolio value over time with trade markers
          </CardDescription>
        </CardHeader>
        <CardContent>
          <PlotlyChart 
            equityCurve={equity_curve} 
            trades={trades}
            ticker={results.params_used.ticker}
          />
        </CardContent>
      </Card>

      {/* Trade Summary */}
      <Card>
        <CardHeader>
          <CardTitle>Trade Summary</CardTitle>
          <CardDescription>
            Recent trades from the backtest
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {trades.slice(-10).map((trade, index) => (
              <div 
                key={index}
                className="flex justify-between items-center p-2 bg-gray-50 rounded"
              >
                <div>
                  <span className="font-medium">{trade.date}</span>
                  <span className={`ml-2 px-2 py-1 rounded text-xs ${
                    trade.side === 'long' 
                      ? 'bg-green-100 text-green-800' 
                      : 'bg-gray-100 text-gray-800'
                  }`}>
                    {trade.side}
                  </span>
                </div>
                <div className="text-sm text-gray-600">
                  ${trade.price.toFixed(2)} × {trade.quantity.toFixed(0)}
                </div>
              </div>
            ))}
            {trades.length === 0 && (
              <p className="text-gray-500 text-center py-4">
                No trades generated for this strategy
              </p>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
