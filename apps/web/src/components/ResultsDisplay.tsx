'use client';

import { BacktestResponse } from '@/lib/types';
import { Card, CardContent } from '@/components/ui/card';
import { PlotlyChart } from '@/components/PlotlyChart';

interface ResultsDisplayProps {
  results: BacktestResponse;
}

const STRATEGY_LABELS: Record<string, string> = {
  ema_crossover: 'EMA Crossover',
  rsi_mean_reversion: 'RSI Mean Reversion',
  sma_crossover: 'SMA Crossover',
  bollinger_breakout: 'Bollinger Breakout',
  momentum: 'Momentum',
};

function PanelHeader({ title, note }: { title: string; note?: string }) {
  return (
    <div className="flex items-center justify-between border-b border-border px-4 py-3">
      <span className="label-eyebrow">{title}</span>
      {note && (
        <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
          {note}
        </span>
      )}
    </div>
  );
}

function Stat({
  label,
  value,
  tone = 'neutral',
}: {
  label: string;
  value: string;
  tone?: 'neutral' | 'positive' | 'negative';
}) {
  const toneClass =
    tone === 'positive' ? 'text-positive' : tone === 'negative' ? 'text-negative' : 'text-foreground';
  return (
    <div className="flex flex-col gap-1 border-b border-r border-border px-4 py-3.5">
      <span className="label-eyebrow">{label}</span>
      <span className={`font-mono text-xl font-medium tabular-nums ${toneClass}`}>{value}</span>
    </div>
  );
}

export function ResultsDisplay({ results }: ResultsDisplayProps) {
  const { metrics, equity_curve, benchmark_curve, trades, params_used } = results;

  const pct = (value: number) => `${(value * 100).toFixed(2)}%`;
  const num = (value: number) => value.toFixed(2);
  const signed = (value: number) => `${value >= 0 ? '+' : ''}${(value * 100).toFixed(2)}%`;

  const cagr = metrics.cagr ?? 0;
  const sharpe = metrics.sharpe ?? 0;
  const maxDrawdown = metrics.max_drawdown ?? 0;
  const winRate = metrics.win_rate ?? 0;
  const totalTrades = metrics.total_trades ?? 0;
  const avgTrade = metrics.avg_trade_return ?? 0;
  const volatility = metrics.volatility ?? 0;

  const strategyLabel = STRATEGY_LABELS[params_used.strategy] ?? params_used.strategy;

  return (
    <div className="space-y-5">
      {/* Run summary */}
      <Card>
        <div className="flex flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3">
          <div className="flex items-baseline gap-2">
            <span className="font-mono text-lg font-semibold tracking-widest text-foreground">
              {params_used.ticker}
            </span>
            <span className="label-eyebrow">{strategyLabel}</span>
          </div>
          <span className="font-mono text-xs text-muted-foreground">
            {params_used.start} to {params_used.end}
          </span>
          <span className="ml-auto font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
            Fees {params_used.fees_bps}bps | Slip {params_used.slip_bps}bps
            {params_used.vol_target ? ` | VolTgt ${pct(params_used.target_vol)}` : ''}
          </span>
        </div>
      </Card>

      {/* Metric board */}
      <Card className="overflow-hidden">
        <PanelHeader title="Performance" note="Strategy" />
        {/* Negative right/bottom border on the grid is hidden by overflow + the
            container border, producing a clean ledger look. */}
        <div className="-mb-px -mr-px grid grid-cols-2 sm:grid-cols-4">
          <Stat label="CAGR" value={pct(cagr)} tone={cagr >= 0 ? 'positive' : 'negative'} />
          <Stat label="Sharpe" value={num(sharpe)} />
          <Stat label="Max Drawdown" value={pct(maxDrawdown)} tone="negative" />
          <Stat label="Win Rate" value={pct(winRate)} />
          <Stat label="Total Trades" value={String(totalTrades)} />
          <Stat
            label="Avg Trade"
            value={signed(avgTrade)}
            tone={avgTrade >= 0 ? 'positive' : 'negative'}
          />
          <Stat label="Volatility" value={pct(volatility)} />
          <Stat label="Universe" value={params_used.ticker} />
        </div>
      </Card>

      {/* Equity curve */}
      <Card>
        <PanelHeader title="Equity Curve" note="Strategy vs Buy & Hold" />
        <CardContent className="p-4">
          <PlotlyChart
            equityCurve={equity_curve}
            benchmarkCurve={benchmark_curve}
            trades={trades}
            ticker={params_used.ticker}
          />
        </CardContent>
      </Card>

      {/* Trade blotter */}
      <Card>
        <PanelHeader title="Trade Blotter" note={`${trades.length} fills`} />
        <CardContent className="p-0">
          {trades.length === 0 ? (
            <p className="px-4 py-8 text-center font-mono text-xs text-muted-foreground">
              No trades generated for this strategy
            </p>
          ) : (
            <div className="max-h-72 overflow-y-auto">
              <table className="w-full border-collapse">
                <thead className="sticky top-0 bg-card">
                  <tr className="label-eyebrow">
                    <th className="border-b border-border px-4 py-2 text-left font-normal">Date</th>
                    <th className="border-b border-border px-4 py-2 text-left font-normal">Side</th>
                    <th className="border-b border-border px-4 py-2 text-right font-normal">Price</th>
                    <th className="border-b border-border px-4 py-2 text-right font-normal">Qty</th>
                  </tr>
                </thead>
                <tbody>
                  {trades
                    .slice(-25)
                    .reverse()
                    .map((trade, index) => (
                      <tr key={index} className="hover:bg-accent/40">
                        <td className="border-b border-border/60 px-4 py-2 font-mono text-xs text-muted-foreground">
                          {trade.date}
                        </td>
                        <td className="border-b border-border/60 px-4 py-2">
                          <span
                            className={`font-mono text-[10px] uppercase tracking-widest ${
                              trade.side === 'long' ? 'text-positive' : 'text-muted-foreground'
                            }`}
                          >
                            {trade.side === 'long' ? 'Long' : 'Flat'}
                          </span>
                        </td>
                        <td className="border-b border-border/60 px-4 py-2 text-right font-mono text-xs tabular-nums">
                          ${trade.price.toFixed(2)}
                        </td>
                        <td className="border-b border-border/60 px-4 py-2 text-right font-mono text-xs tabular-nums text-muted-foreground">
                          {trade.quantity.toFixed(0)}
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
