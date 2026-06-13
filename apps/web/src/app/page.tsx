'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { BacktestRequest, StrategyId } from '@/lib/types';
import { runBacktest } from '@/lib/api';
import { saveParams, loadParams, getDefaultParams } from '@/lib/storage';
import { ResultsDisplay } from '@/components/ResultsDisplay';
import { AboutModal } from '@/components/AboutModal';

const STRATEGIES: { id: StrategyId; label: string }[] = [
  { id: 'ema_crossover', label: 'EMA Crossover' },
  { id: 'rsi_mean_reversion', label: 'RSI Mean Reversion' },
  { id: 'sma_crossover', label: 'SMA Crossover' },
  { id: 'bollinger_breakout', label: 'Bollinger Breakout' },
  { id: 'momentum', label: 'Momentum' },
];

const labelClass = 'label-eyebrow';
const fieldClass = 'font-mono text-sm rounded-sm';

export default function Home() {
  const [params, setParams] = useState<BacktestRequest>(getDefaultParams());
  const [results, setResults] = useState<Awaited<ReturnType<typeof runBacktest>> | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showAbout, setShowAbout] = useState(false);

  useEffect(() => {
    const savedParams = loadParams();
    if (savedParams) {
      setParams(savedParams);
    }
  }, []);

  const handleParamChange = (key: keyof BacktestRequest, value: BacktestRequest[keyof BacktestRequest]) => {
    const newParams = { ...params, [key]: value };
    setParams(newParams);
    saveParams(newParams);
  };

  const handleRunBacktest = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await runBacktest(params);
      setResults(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const numberField = (
    key: keyof BacktestRequest,
    label: string,
    opts: { min?: number; max?: number; step?: number } = {}
  ) => (
    <div className="space-y-1.5">
      <Label htmlFor={key} className={labelClass}>
        {label}
      </Label>
      <Input
        id={key}
        type="number"
        className={fieldClass}
        value={params[key] as number}
        onChange={(e) =>
          handleParamChange(
            key,
            opts.step && opts.step < 1 ? parseFloat(e.target.value) : parseInt(e.target.value, 10)
          )
        }
        min={opts.min}
        max={opts.max}
        step={opts.step}
      />
    </div>
  );

  const renderStrategyParams = () => {
    switch (params.strategy) {
      case 'ema_crossover':
        return (
          <div className="grid grid-cols-2 gap-3">
            {numberField('ema_fast', 'Fast EMA', { min: 2, max: 250 })}
            {numberField('ema_slow', 'Slow EMA', { min: 3, max: 500 })}
          </div>
        );
      case 'rsi_mean_reversion':
        return (
          <div className="grid grid-cols-3 gap-3">
            {numberField('rsi_period', 'RSI Period', { min: 2, max: 50 })}
            {numberField('oversold', 'Oversold', { min: 5, max: 45 })}
            {numberField('overbought', 'Overbought', { min: 55, max: 95 })}
          </div>
        );
      case 'sma_crossover':
        return (
          <div className="grid grid-cols-2 gap-3">
            {numberField('sma_fast', 'Fast SMA', { min: 2, max: 250 })}
            {numberField('sma_slow', 'Slow SMA', { min: 3, max: 500 })}
          </div>
        );
      case 'bollinger_breakout':
        return (
          <div className="grid grid-cols-2 gap-3">
            {numberField('bb_window', 'BB Window', { min: 5, max: 100 })}
            {numberField('bb_std', 'BB Std Dev', { min: 1, max: 3, step: 0.1 })}
          </div>
        );
      case 'momentum':
        return (
          <div className="grid grid-cols-2 gap-3">
            {numberField('lookback', 'Lookback', { min: 2, max: 100 })}
            {numberField('threshold', 'Threshold', { min: 0, max: 0.5, step: 0.01 })}
          </div>
        );
      default:
        return null;
    }
  };

  const activeStrategy = STRATEGIES.find((s) => s.id === params.strategy)?.label ?? '';

  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* Status bar */}
      <header className="sticky top-0 z-30 border-b border-border bg-background/90 backdrop-blur supports-[backdrop-filter]:bg-background/70">
        <div className="mx-auto flex h-12 max-w-[1500px] items-center justify-between px-5">
          <div className="flex items-center gap-3">
            <div className="h-4 w-4 bg-primary" />
            <span className="font-mono text-sm font-semibold tracking-[0.2em] text-foreground">
              QUANT<span className="text-primary">{'//'}</span>TERMINAL
            </span>
            <span className="hidden font-mono text-[10px] uppercase tracking-widest text-muted-foreground sm:inline">
              Multi-Strategy Backtester
            </span>
          </div>
          <div className="flex items-center gap-4">
            <span className="hidden items-center gap-2 font-mono text-[10px] uppercase tracking-widest text-muted-foreground md:flex">
              <span className="h-1.5 w-1.5 rounded-full bg-positive" />
              Market Data
            </span>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowAbout(true)}
              className="rounded-sm font-mono text-[11px] uppercase tracking-widest text-muted-foreground hover:text-foreground"
            >
              About
            </Button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-[1500px] px-5 py-6">
        <div className="grid grid-cols-1 gap-5 lg:grid-cols-[minmax(340px,380px)_1fr]">
          {/* Control rail */}
          <Card className="h-fit lg:sticky lg:top-[4.5rem]">
            <div className="flex items-center justify-between border-b border-border px-4 py-3">
              <span className="label-eyebrow">Configuration</span>
              <span className="font-mono text-[10px] uppercase tracking-widest text-primary">
                {activeStrategy}
              </span>
            </div>
            <CardContent className="space-y-5 p-4">
              {/* Strategy */}
              <div className="space-y-1.5">
                <Label htmlFor="strategy" className={labelClass}>
                  Strategy
                </Label>
                <select
                  id="strategy"
                  value={params.strategy}
                  onChange={(e) => handleParamChange('strategy', e.target.value as StrategyId)}
                  className="flex h-10 w-full rounded-sm border border-input bg-background px-3 py-2 font-mono text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                >
                  {STRATEGIES.map((s) => (
                    <option key={s.id} value={s.id}>
                      {s.label}
                    </option>
                  ))}
                </select>
              </div>

              {/* Ticker */}
              <div className="space-y-1.5">
                <Label htmlFor="ticker" className={labelClass}>
                  Ticker
                </Label>
                <Input
                  id="ticker"
                  className={`${fieldClass} uppercase tracking-widest`}
                  value={params.ticker}
                  onChange={(e) => handleParamChange('ticker', e.target.value.toUpperCase())}
                  placeholder="SPY"
                />
              </div>

              {/* Date Range */}
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1.5">
                  <Label htmlFor="start" className={labelClass}>
                    Start
                  </Label>
                  <Input
                    id="start"
                    type="date"
                    className={fieldClass}
                    value={params.start}
                    onChange={(e) => handleParamChange('start', e.target.value)}
                  />
                </div>
                <div className="space-y-1.5">
                  <Label htmlFor="end" className={labelClass}>
                    End
                  </Label>
                  <Input
                    id="end"
                    type="date"
                    className={fieldClass}
                    value={params.end}
                    onChange={(e) => handleParamChange('end', e.target.value)}
                  />
                </div>
              </div>

              <div className="h-px bg-border" />

              {/* Strategy-specific parameters */}
              {renderStrategyParams()}

              <div className="h-px bg-border" />

              {/* Trading Costs */}
              <div className="grid grid-cols-2 gap-3">
                {numberField('fees_bps', 'Fees (bps)', { min: 0, max: 100, step: 0.1 })}
                {numberField('slip_bps', 'Slippage (bps)', { min: 0, max: 100, step: 0.1 })}
              </div>

              {/* Volatility Targeting */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <Label htmlFor="vol_target" className={labelClass}>
                    Volatility Targeting
                  </Label>
                  <Switch
                    id="vol_target"
                    size="sm"
                    checked={params.vol_target}
                    onChange={(e) => handleParamChange('vol_target', e.target.checked)}
                  />
                </div>

                {params.vol_target &&
                  numberField('target_vol', 'Target Volatility', { min: 0.01, max: 1, step: 0.01 })}
              </div>

              {/* Run Button */}
              <Button
                onClick={handleRunBacktest}
                disabled={loading}
                className="w-full rounded-sm font-mono text-xs font-semibold uppercase tracking-[0.2em]"
                size="lg"
              >
                {loading ? 'Running...' : 'Run Backtest'}
              </Button>

              {error && (
                <div className="rounded-sm border border-negative/40 bg-negative/10 p-3">
                  <p className="font-mono text-xs text-negative">{error}</p>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Results */}
          <div>
            {results ? (
              <ResultsDisplay results={results} />
            ) : (
              <Card className="terminal-grid flex h-[460px] items-center justify-center">
                <div className="text-center">
                  <p className="font-mono text-sm uppercase tracking-[0.25em] text-muted-foreground">
                    No Run Loaded
                  </p>
                  <p className="mt-3 font-mono text-xs text-muted-foreground/70">
                    Configure parameters and execute a backtest
                    <span className="ml-1 inline-block h-3 w-1.5 translate-y-0.5 animate-pulse bg-primary" />
                  </p>
                </div>
              </Card>
            )}
          </div>
        </div>
      </main>

      <AboutModal open={showAbout} onOpenChange={setShowAbout} />
    </div>
  );
}
