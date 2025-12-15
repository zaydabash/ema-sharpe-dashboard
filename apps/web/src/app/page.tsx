'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { BacktestRequest, BacktestResponse } from '@/lib/types';
import { runBacktest } from '@/lib/api';
import { saveParams, loadParams, getDefaultParams } from '@/lib/storage';
import { ResultsDisplay } from '@/components/ResultsDisplay';
import { AboutModal } from '@/components/AboutModal';

export default function Home() {
  const [params, setParams] = useState<BacktestRequest>(getDefaultParams());
  const [results, setResults] = useState<BacktestResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showAbout, setShowAbout] = useState(false);

  useEffect(() => {
    const savedParams = loadParams();
    if (savedParams) {
      setParams(savedParams);
    }
  }, []);

  const handleParamChange = (key: keyof BacktestRequest, value: any) => {
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

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            EMA + Sharpe Dashboard
          </h1>
          <p className="text-lg text-gray-600 mb-6">
            Interactive EMA crossover backtest with realistic costs and volatility targeting
          </p>
          <Button 
            variant="outline" 
            onClick={() => setShowAbout(true)}
            className="mb-8"
          >
            About This Tool
          </Button>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Parameters Form */}
          <Card>
            <CardHeader>
              <CardTitle>Backtest Parameters</CardTitle>
              <CardDescription>
                Configure your EMA crossover strategy parameters
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Ticker */}
              <div className="space-y-2">
                <Label htmlFor="ticker">Ticker Symbol</Label>
                <Input
                  id="ticker"
                  value={params.ticker}
                  onChange={(e) => handleParamChange('ticker', e.target.value.toUpperCase())}
                  placeholder="SPY"
                />
              </div>

              {/* Date Range */}
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="start">Start Date</Label>
                  <Input
                    id="start"
                    type="date"
                    value={params.start}
                    onChange={(e) => handleParamChange('start', e.target.value)}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="end">End Date</Label>
                  <Input
                    id="end"
                    type="date"
                    value={params.end}
                    onChange={(e) => handleParamChange('end', e.target.value)}
                  />
                </div>
              </div>

              {/* EMA Periods */}
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="ema_fast">Fast EMA</Label>
                  <Input
                    id="ema_fast"
                    type="number"
                    value={params.ema_fast}
                    onChange={(e) => handleParamChange('ema_fast', parseInt(e.target.value))}
                    min="2"
                    max="250"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="ema_slow">Slow EMA</Label>
                  <Input
                    id="ema_slow"
                    type="number"
                    value={params.ema_slow}
                    onChange={(e) => handleParamChange('ema_slow', parseInt(e.target.value))}
                    min="2"
                    max="500"
                  />
                </div>
              </div>

              {/* Trading Costs */}
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="fees_bps">Fees (bps)</Label>
                  <Input
                    id="fees_bps"
                    type="number"
                    value={params.fees_bps}
                    onChange={(e) => handleParamChange('fees_bps', parseFloat(e.target.value))}
                    min="0"
                    max="100"
                    step="0.1"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="slip_bps">Slippage (bps)</Label>
                  <Input
                    id="slip_bps"
                    type="number"
                    value={params.slip_bps}
                    onChange={(e) => handleParamChange('slip_bps', parseFloat(e.target.value))}
                    min="0"
                    max="100"
                    step="0.1"
                  />
                </div>
              </div>

              {/* Volatility Targeting */}
              <div className="space-y-4">
                <div className="flex items-center space-x-2">
                  <Switch
                    id="vol_target"
                    checked={params.vol_target}
                    onCheckedChange={(checked) => handleParamChange('vol_target', checked)}
                  />
                  <Label htmlFor="vol_target">Enable Volatility Targeting</Label>
                </div>
                
                {params.vol_target && (
                  <div className="space-y-2">
                    <Label htmlFor="target_vol">Target Volatility</Label>
                    <Input
                      id="target_vol"
                      type="number"
                      value={params.target_vol}
                      onChange={(e) => handleParamChange('target_vol', parseFloat(e.target.value))}
                      min="0.01"
                      max="1"
                      step="0.01"
                    />
                  </div>
                )}
              </div>

              {/* Run Button */}
              <Button 
                onClick={handleRunBacktest} 
                disabled={loading}
                className="w-full"
                size="lg"
              >
                {loading ? 'Running Backtest...' : 'Run Backtest'}
              </Button>

              {error && (
                <div className="p-4 bg-red-50 border border-red-200 rounded-md">
                  <p className="text-red-800 text-sm">{error}</p>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Results */}
          <div>
            {results ? (
              <ResultsDisplay results={results} />
            ) : (
              <Card>
                <CardContent className="flex items-center justify-center h-96">
                  <div className="text-center text-gray-500">
                    <p className="text-lg mb-2">No results yet</p>
                    <p className="text-sm">Configure parameters and run a backtest to see results</p>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>

      <AboutModal open={showAbout} onOpenChange={setShowAbout} />
    </div>
  );
}
