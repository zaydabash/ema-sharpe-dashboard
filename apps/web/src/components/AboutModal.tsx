'use client';

import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';

interface AboutModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <h3 className="label-eyebrow mb-2">{title}</h3>
      <div className="text-sm leading-relaxed text-muted-foreground">{children}</div>
    </div>
  );
}

export function AboutModal({ open, onOpenChange }: AboutModalProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[82vh] max-w-2xl overflow-y-auto rounded-sm bg-card">
        <DialogHeader>
          <DialogTitle className="font-mono uppercase tracking-[0.15em]">
            About Quant Terminal
          </DialogTitle>
          <DialogDescription>
            A backtesting workbench for five systematic strategies with realistic costs and a
            buy-and-hold benchmark.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6 pt-2">
          <Section title="Strategies">
            <ul className="space-y-1.5">
              <li>
                <strong className="text-foreground">EMA / SMA Crossover</strong>: trend following on
                fast/slow moving-average crosses.
              </li>
              <li>
                <strong className="text-foreground">RSI Mean Reversion</strong>: buy oversold, exit
                on overbought.
              </li>
              <li>
                <strong className="text-foreground">Bollinger Breakout</strong>: enter on breaks
                above the upper band.
              </li>
              <li>
                <strong className="text-foreground">Momentum</strong>: go long when trailing return
                clears a threshold.
              </li>
            </ul>
          </Section>

          <Section title="Engine">
            <ul className="space-y-1.5">
              <li>
                <strong className="text-foreground">Realistic costs:</strong> configurable fees and
                slippage on every fill.
              </li>
              <li>
                <strong className="text-foreground">Volatility targeting:</strong> optional daily
                exposure scaling (rolling 20-day vol, capped at 2x leverage).
              </li>
              <li>
                <strong className="text-foreground">Benchmark:</strong> every run is compared against
                buy-and-hold.
              </li>
            </ul>
          </Section>

          <Section title="Metrics">
            <ul className="space-y-1.5">
              <li>
                <strong className="text-foreground">CAGR</strong>: annualized compound return.
              </li>
              <li>
                <strong className="text-foreground">Sharpe</strong>: risk-adjusted return (higher is
                better).
              </li>
              <li>
                <strong className="text-foreground">Max Drawdown</strong>: largest peak-to-trough
                decline.
              </li>
              <li>
                <strong className="text-foreground">Win Rate / Volatility</strong>: share of
                profitable trades and annualized standard deviation.
              </li>
            </ul>
          </Section>

          <div className="rounded-sm border border-primary/30 bg-primary/5 p-4">
            <h3 className="label-eyebrow mb-2 text-primary">Disclaimer</h3>
            <p className="text-sm leading-relaxed text-muted-foreground">
              For educational purposes only. Past performance does not guarantee future results, and
              backtests may not reflect live trading conditions. Always do your own research before
              making investment decisions.
            </p>
          </div>

          <div className="flex justify-end border-t border-border pt-4">
            <Button
              onClick={() => onOpenChange(false)}
              className="rounded-sm font-mono text-[11px] uppercase tracking-widest"
            >
              Close
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
