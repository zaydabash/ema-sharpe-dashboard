'use client';

import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';

interface AboutModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function AboutModal({ open, onOpenChange }: AboutModalProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>About EMA + Sharpe Dashboard</DialogTitle>
          <DialogDescription>
            Learn about the EMA crossover strategy and how this tool works
          </DialogDescription>
        </DialogHeader>
        
        <div className="space-y-6 text-sm">
          <div>
            <h3 className="font-semibold text-base mb-2">What is EMA Crossover?</h3>
            <p className="text-gray-600 mb-2">
              The Exponential Moving Average (EMA) crossover strategy is a popular trend-following approach:
            </p>
            <ul className="list-disc list-inside text-gray-600 space-y-1 ml-4">
              <li>Go <strong>long</strong> when the fast EMA crosses above the slow EMA</li>
              <li>Go <strong>flat</strong> when the fast EMA crosses below the slow EMA</li>
              <li>EMAs give more weight to recent prices, making them more responsive than simple moving averages</li>
            </ul>
          </div>

          <div>
            <h3 className="font-semibold text-base mb-2">Key Features</h3>
            <ul className="list-disc list-inside text-gray-600 space-y-1 ml-4">
              <li><strong>Realistic Costs:</strong> Configurable fees and slippage applied to all trades</li>
              <li><strong>Volatility Targeting:</strong> Optional daily exposure scaling to target volatility</li>
              <li><strong>Performance Metrics:</strong> CAGR, Sharpe ratio, max drawdown, win rate</li>
              <li><strong>Interactive Charts:</strong> Visualize equity curve with trade markers</li>
            </ul>
          </div>

          <div>
            <h3 className="font-semibold text-base mb-2">Understanding the Metrics</h3>
            <ul className="list-disc list-inside text-gray-600 space-y-1 ml-4">
              <li><strong>CAGR:</strong> Compound Annual Growth Rate - annualized return</li>
              <li><strong>Sharpe Ratio:</strong> Risk-adjusted return (higher is better)</li>
              <li><strong>Max Drawdown:</strong> Largest peak-to-trough decline</li>
              <li><strong>Win Rate:</strong> Percentage of profitable trades</li>
              <li><strong>Volatility:</strong> Annualized standard deviation of returns</li>
            </ul>
          </div>

          <div>
            <h3 className="font-semibold text-base mb-2">Volatility Targeting</h3>
            <p className="text-gray-600 mb-2">
              When enabled, the strategy scales position size daily to target a specific volatility level:
            </p>
            <ul className="list-disc list-inside text-gray-600 space-y-1 ml-4">
              <li>Calculates rolling 20-day volatility</li>
              <li>Scales exposure: target_vol / current_vol</li>
              <li>Caps leverage at 2x for safety</li>
              <li>Helps maintain consistent risk levels</li>
            </ul>
          </div>

          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
            <h3 className="font-semibold text-base mb-2 text-yellow-800">Important Disclaimers</h3>
            <ul className="list-disc list-inside text-yellow-700 space-y-1 ml-4">
              <li>This tool is for <strong>educational purposes only</strong></li>
              <li>Past performance does not guarantee future results</li>
              <li>Backtesting results may not reflect real trading conditions</li>
              <li>Consider transaction costs, market impact, and liquidity</li>
              <li>Always do your own research before making investment decisions</li>
            </ul>
          </div>

          <div className="text-center">
            <Button onClick={() => onOpenChange(false)}>
              Got it!
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
