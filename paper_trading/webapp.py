from flask import Flask, render_template_string, jsonify, send_file
import threading
import io
import pandas as pd
import json
import time

app = Flask(__name__)

# Shared state for the web app
state = {
    'capital': 1_000_000,
    'trade_log': [],
    'open_trades': [],
    'live_price': None,
    'last_update': time.time()
}

def update_state(account, trade_result):
    state['capital'] = account.get_capital()
    state['trade_log'] = account.get_trade_log()
    state['open_trades'] = account.get_open_trades()
    state['last_update'] = time.time()

def set_live_price(price):
    state['live_price'] = price
    state['last_update'] = time.time()

@app.route('/data')
def data():
    # Add timestamp to help with debugging
    return jsonify({
        'capital': state['capital'],
        'trade_log': state['trade_log'][-100:],
        'open_trades': state['open_trades'],
        'live_price': state['live_price'],
        'last_update': state['last_update']
    })

@app.route('/download_trades')
def download_trades():
    df = pd.DataFrame(state['trade_log'])
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(io.BytesIO(buf.getvalue().encode()), mimetype='text/csv', as_attachment=True, download_name='trade_log.csv')

@app.route('/')
def dashboard():
    html = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Quant Trading Terminal</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body { background: #181c20; color: #e0e0e0; font-family: 'Fira Mono', 'Consolas', monospace; }
            .navbar { background: #101214 !important; }
            .card { background: #23272b; border: none; box-shadow: 0 2px 16px #000a 0px 2px 8px #00ffe1a0; }
            .card-title { color: #00ffe1; letter-spacing: 1px; }
            .glow { color: #00ffe1; text-shadow: 0 0 8px #00ffe1, 0 0 2px #00ffe1; }
            .ticker { font-size: 2.2rem; font-weight: bold; letter-spacing: 2px; }
            .ticker-up { color: #00ff99; text-shadow: 0 0 8px #00ff99; }
            .ticker-down { color: #ff0055; text-shadow: 0 0 8px #ff0055; }
            .table { background: #23272b; color: #e0e0e0; }
            .table th, .table td { border-color: #333; }
            .btn-outline-secondary { border-color: #00ffe1; color: #00ffe1; }
            .btn-outline-secondary:hover { background: #00ffe1; color: #181c20; }
            .chartjs-render-monitor { background: #181c20 !important; }
            .chart-dark { background: #181c20 !important; border-radius: 8px; }
            .card { margin-bottom: 1rem; }
            .table { font-size: 0.95rem; }
            .scrollable { max-height: 350px; overflow-y: auto; }
            .status { font-size: 0.8rem; color: #666; }
        </style>
        <link href="https://fonts.googleapis.com/css2?family=Fira+Mono:wght@400;700&display=swap" rel="stylesheet">
    </head>
    <body>
    <nav class="navbar navbar-expand-lg navbar-dark mb-4">
      <div class="container-fluid">
        <a class="navbar-brand glow" href="#">QUANT TRADING TERMINAL</a>
        <span class="status" id="status">Connecting...</span>
      </div>
    </nav>
    <div class="container-fluid">
      <div class="row mb-3">
        <div class="col-md-4">
          <div class="card text-white" style="background: #23272b;">
            <div class="card-body">
              <h5 class="card-title">Live BTC Price</h5>
              <div class="ticker" id="live_price">$N/A</div>
            </div>
          </div>
        </div>
        <div class="col-md-4">
          <div class="card text-white" style="background: #23272b;">
            <div class="card-body">
              <h5 class="card-title">Account Balance</h5>
              <div class="ticker glow" id="capital">$0.00</div>
            </div>
          </div>
        </div>
        <div class="col-md-4 d-flex align-items-center">
          <a href="/download_trades" class="btn btn-outline-secondary w-100">Download Trade Log (CSV)</a>
        </div>
      </div>
      <div class="row mb-3">
        <div class="col-md-8">
          <div class="card">
            <div class="card-body">
              <h5 class="card-title">Equity Curve</h5>
              <canvas id="equityCurve" class="chart-dark"></canvas>
            </div>
          </div>
        </div>
        <div class="col-md-4">
          <div class="card">
            <div class="card-body">
              <h5 class="card-title">Trade Reason Distribution</h5>
              <canvas id="reasonPie" class="chart-dark"></canvas>
            </div>
          </div>
          <div class="card mt-3">
            <div class="card-body">
              <h5 class="card-title">PnL per Trade</h5>
              <canvas id="pnlBar" class="chart-dark"></canvas>
            </div>
          </div>
        </div>
      </div>
      <div class="row mb-3">
        <div class="col-12">
          <div class="card">
            <div class="card-body scrollable">
              <h5 class="card-title">Open Positions</h5>
              <div id="openPositions"></div>
            </div>
          </div>
        </div>
      </div>
      <div class="row mb-3">
        <div class="col-12">
          <div class="card">
            <div class="card-body scrollable">
              <h5 class="card-title">Trade Log (last 100)</h5>
              <div id="tradeLog"></div>
            </div>
          </div>
        </div>
      </div>
    </div>
    <script>
    let equityChart, reasonChart, pnlChart;
    let lastPrice = null;
    let lastUpdate = null;
    
    function updateStatus(data) {
        const now = Date.now() / 1000;
        const statusElem = document.getElementById('status');
        if (data.last_update) {
            const age = now - data.last_update;
            if (age < 5) {
                statusElem.textContent = 'Connected';
                statusElem.style.color = '#00ff99';
            } else {
                statusElem.textContent = `Last update ${Math.round(age)}s ago`;
                statusElem.style.color = '#ff0055';
            }
        }
    }
    
    function updateDashboard() {
      fetch('/data').then(r => r.json()).then(data => {
        updateStatus(data);
        
        // Live price ticker with up/down color
        let priceElem = document.getElementById('live_price');
        if (data.live_price) {
          let price = Number(data.live_price);
          let priceStr = `$${price.toLocaleString(undefined, {minimumFractionDigits:2, maximumFractionDigits:2})}`;
          if (lastPrice !== null) {
            if (price > lastPrice) {
              priceElem.className = 'ticker ticker-up';
            } else if (price < lastPrice) {
              priceElem.className = 'ticker ticker-down';
            } else {
              priceElem.className = 'ticker';
            }
          }
          priceElem.innerText = priceStr;
          lastPrice = price;
        } else {
          priceElem.className = 'ticker';
          priceElem.innerText = 'N/A';
        }
        
        document.getElementById('capital').innerText = `$${Number(data.capital).toLocaleString(undefined, {minimumFractionDigits:2, maximumFractionDigits:2})}`;
        
        // Equity curve
        let equity = [data.capital];
        if (data.trade_log && data.trade_log.length > 0) {
          equity = [data.trade_log[0].capital];
          for (let i = 1; i < data.trade_log.length; i++) {
            equity.push(data.trade_log[i].capital);
          }
        }
        let labels = equity.map((_, i) => i+1);
        if (!equityChart) {
          equityChart = new Chart(document.getElementById('equityCurve').getContext('2d'), {
            type: 'line',
            data: { labels: labels, datasets: [{ label: 'Equity', data: equity, borderColor: '#00ffe1', backgroundColor: 'rgba(0,255,225,0.1)', fill: true, tension: 0.2, pointRadius: 0 }] },
            options: { responsive: true, plugins: { legend: { display: false } }, scales: { x: { ticks: { color: '#00ffe1' } }, y: { ticks: { color: '#00ffe1' } } }, backgroundColor: '#181c20' }
          });
        } else {
          equityChart.data.labels = labels;
          equityChart.data.datasets[0].data = equity;
          equityChart.update();
        }
        
        // PnL per trade
        if (data.trade_log && data.trade_log.length > 0) {
          let pnls = data.trade_log.map(t => t.net_pnl || 0);
          let pnlLabels = data.trade_log.map((t, i) => i+1);
          if (!pnlChart) {
            pnlChart = new Chart(document.getElementById('pnlBar').getContext('2d'), {
              type: 'bar',
              data: { labels: pnlLabels, datasets: [{ label: 'PnL', data: pnls, backgroundColor: '#00ff99' }] },
              options: { responsive: true, plugins: { legend: { display: false } }, scales: { x: { ticks: { color: '#00ff99' } }, y: { ticks: { color: '#00ff99' } } }, backgroundColor: '#181c20' }
            });
          } else {
            pnlChart.data.labels = pnlLabels;
            pnlChart.data.datasets[0].data = pnls;
            pnlChart.update();
          }
          
          // Trade reason pie
          let reasonCounts = {};
          data.trade_log.forEach(t => { reasonCounts[t.reason] = (reasonCounts[t.reason]||0)+1; });
          let reasonLabels = Object.keys(reasonCounts);
          let reasonData = Object.values(reasonCounts);
          if (!reasonChart) {
            reasonChart = new Chart(document.getElementById('reasonPie').getContext('2d'), {
              type: 'pie',
              data: { labels: reasonLabels, datasets: [{ data: reasonData, backgroundColor: ['#00ffe1','#ff0055','#ffc107','#00ff99','#6c757d','#17a2b8'] }] },
              options: { responsive: true, plugins: { legend: { labels: { color: '#00ffe1' } } } }
            });
          } else {
            reasonChart.data.labels = reasonLabels;
            reasonChart.data.datasets[0].data = reasonData;
            reasonChart.update();
          }
        }
        
        // Open positions table
        let openHtml = '<table class="table table-sm table-dark table-striped"><thead><tr>';
        if (data.open_trades && data.open_trades.length > 0) {
          Object.keys(data.open_trades[0]).forEach(k => { openHtml += `<th>${k}</th>`; });
          openHtml += '</tr></thead><tbody>';
          data.open_trades.forEach(pos => {
            openHtml += '<tr>';
            Object.values(pos).forEach(v => { openHtml += `<td>${v}</td>`; });
            openHtml += '</tr>';
          });
          openHtml += '</tbody></table>';
        } else {
          openHtml = '<p>No open positions.</p>';
        }
        document.getElementById('openPositions').innerHTML = openHtml;
        
        // Trade log table
        let logHtml = '<table class="table table-sm table-dark table-striped"><thead><tr>';
        if (data.trade_log && data.trade_log.length > 0) {
          Object.keys(data.trade_log[0]).forEach(k => { logHtml += `<th>${k}</th>`; });
          logHtml += '</tr></thead><tbody>';
          data.trade_log.forEach(trade => {
            logHtml += '<tr>';
            Object.values(trade).forEach(v => { logHtml += `<td>${v}</td>`; });
            logHtml += '</tr>';
          });
          logHtml += '</tbody></table>';
        } else {
          logHtml = '<p>No trades yet.</p>';
        }
        document.getElementById('tradeLog').innerHTML = logHtml;
      });
    }
    
    // Update every 2 seconds
    setInterval(updateDashboard, 2000);
    updateDashboard();
    </script>
    </body>
    </html>
    '''
    return render_template_string(html)

def run_webapp():
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

if __name__ == "__main__":
    run_webapp() 