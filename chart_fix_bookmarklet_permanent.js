// Permanent Chart Fix Bookmarklet
// Save this as a bookmark and click it whenever you visit the site

javascript:(function(){
  // Store the fix function globally so it persists
  window.fixAllCharts = function() {
    let data = null;
    
    // Try different ways the data might be stored
    if (window.lastResponseData) {
      data = window.lastResponseData;
    } else if (window.responseData) {
      data = window.responseData;
    } else if (window.backtestData) {
      data = window.backtestData;
    } else {
      console.log('No stored data found. Please run a backtest first.');
      return;
    }
    
    if (data && data.equity_curve) {
      // 1. EQUITY CHART
      const equityData = data.equity_curve.map(point => ({
        x: new Date(point.date),
        y: point.equity
      }));
      
      Plotly.newPlot('chart_equity', [{
        x: equityData.map(p => p.x),
        y: equityData.map(p => p.y),
        mode: 'lines',
        line: { width: 2, color: '#3b82f6' },
        name: 'Strategy'
      }], {
        title: 'Equity Curve',
        xaxis: { title: 'Date', type: 'date' },
        yaxis: { title: 'Equity Value', type: 'linear' },
        margin: { t: 40, r: 20, b: 40, l: 60 }
      }, { responsive: true });
      
      // 2. DRAWDOWN CHART
      const drawdownData = data.equity_curve.map(point => {
        const peak = Math.max(...data.equity_curve.slice(0, data.equity_curve.indexOf(point) + 1).map(p => p.equity));
        return {
          x: new Date(point.date),
          y: (point.equity - peak) / peak
        };
      });
      
      Plotly.newPlot('chart_dd', [{
        x: drawdownData.map(p => p.x),
        y: drawdownData.map(p => p.y),
        mode: 'lines',
        line: { width: 2, color: '#ef4444' },
        name: 'Drawdown',
        fill: 'tonexty'
      }], {
        title: 'Drawdown',
        xaxis: { title: 'Date', type: 'date' },
        yaxis: { title: 'Drawdown %', type: 'linear', tickformat: '.1%' },
        margin: { t: 40, r: 20, b: 40, l: 60 }
      }, { responsive: true });
      
      // 3. DAILY EXPOSURE CHART (if available)
      if (data.exposure) {
        const exposureData = data.exposure.map(point => ({
          x: new Date(point.date),
          y: point.exposure
        }));
        
        Plotly.newPlot('chart_expo', [{
          x: exposureData.map(p => p.x),
          y: exposureData.map(p => p.y),
          mode: 'lines',
          line: { width: 2, color: '#10b981' },
          name: 'Exposure'
        }], {
          title: 'Daily Exposure',
          xaxis: { title: 'Date', type: 'date' },
          yaxis: { title: 'Exposure', type: 'linear', range: [0, 1] },
          margin: { t: 40, r: 20, b: 40, l: 60 }
        }, { responsive: true });
      }
      
      console.log('All charts fixed!');
    }
  };
  
  // Auto-run the fix if data is available
  if (window.lastResponseData || window.responseData || window.backtestData) {
    window.fixAllCharts();
  } else {
    console.log('Chart fix loaded! Run a backtest, then click this bookmark again to fix charts.');
  }
})();
