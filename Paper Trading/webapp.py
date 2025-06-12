from flask import Flask, render_template_string, jsonify, send_file
import threading
import io
import pandas as pd
import json

app = Flask(__name__)

# Shared state for the web app
state = {
    'capital': 1_000_000,
    'trade_log': [],
    'open_trades': [],
    'live_price': None
}

def update_state(account, trade_result):
    state['capital'] = account.get_capital()
    state['trade_log'] = account.get_trade_log()
    state['open_trades'] = account.get_open_trades()

def set_live_price(price):
    state['live_price'] = price

@app.route('/data')
def data():
    # Serve all state as JSON for JS auto-refresh
    return jsonify({
        'capital': state['capital'],
        'trade_log': state['trade_log'][-100:],
        'open_trades': state['open_trades'],
        'live_price': state['live_price']
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
        <title>Paper Trading Dashboard</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body { background: #f8f9fa; }
            .card { margin-bottom: 1rem; }
            .table { font-size: 0.95rem; }
        </style>
    </head>
    <body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark mb-4">
      <div class="container-fluid">
        <a class="navbar-brand" href="#">Paper Trading Dashboard</a>
      </div>
    </nav>
    <div class="container">
      <div class="row mb-3">
        <div class="col-md-4">
          <div class="card text-white bg-primary">
            <div class="card-body">
              <h5 class="card-title">Live BTC Price</h5>
              <h3 id="live_price">$N/A</h3>
            </div>
          </div>
        </div>
        <div class="col-md-4">
          <div class="card text-white bg-success">
            <div class="card-body">
              <h5 class="card-title">Account Balance</h5>
              <h3 id="capital">$0.00</h3>
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
              <canvas id="equityCurve"></canvas>
            </div>
          </div>
        </div>
        <div class="col-md-4">
          <div class="card">
            <div class="card-body">
              <h5 class="card-title">Trade Reason Distribution</h5>
              <canvas id="reasonPie"></canvas>
            </div>
          </div>
          <div class="card mt-3">
            <div class="card-body">
              <h5 class="card-title">PnL per Trade</h5>
              <canvas id="pnlBar"></canvas>
            </div>
          </div>
        </div>
      </div>
      <div class="row mb-3">
        <div class="col-12">
          <div class="card">
            <div class="card-body">
              <h5 class="card-title">Open Positions</h5>
              <div id="openPositions"></div>
            </div>
          </div>
        </div>
      </div>
      <div class="row mb-3">
        <div class="col-12">
          <div class="card">
            <div class="card-body">
              <h5 class="card-title">Trade Log (last 100)</h5>
              <div id="tradeLog"></div>
            </div>
          </div>
        </div>
      </div>
    </div>
    <script>
    let equityChart, reasonChart, pnlChart;
    function updateDashboard() {
      fetch('/data').then(r => r.json()).then(data => {
        document.getElementById('live_price').innerText = data.live_price ? `$${Number(data.live_price).toLocaleString(undefined, {minimumFractionDigits:2, maximumFractionDigits:2})}` : 'N/A';
        document.getElementById('capital').innerText = `$${Number(data.capital).toLocaleString(undefined, {minimumFractionDigits:2, maximumFractionDigits:2})}`;
        // Equity curve
        let equity = [data.capital];
        if (data.trade_log.length > 0) {
          equity = [data.trade_log[0].capital];
          for (let i = 1; i < data.trade_log.length; i++) {
            equity.push(data.trade_log[i].capital);
          }
        }
        let labels = equity.map((_, i) => i+1);
        if (!equityChart) {
          equityChart = new Chart(document.getElementById('equityCurve').getContext('2d'), {
            type: 'line',
            data: { labels: labels, datasets: [{ label: 'Equity', data: equity, borderColor: '#007bff', fill: false }] },
            options: { responsive: true, plugins: { legend: { display: false } } }
          });
        } else {
          equityChart.data.labels = labels;
          equityChart.data.datasets[0].data = equity;
          equityChart.update();
        }
        // PnL per trade
        let pnls = data.trade_log.map(t => t.net_pnl || 0);
        let pnlLabels = data.trade_log.map((t, i) => i+1);
        if (!pnlChart) {
          pnlChart = new Chart(document.getElementById('pnlBar').getContext('2d'), {
            type: 'bar',
            data: { labels: pnlLabels, datasets: [{ label: 'PnL', data: pnls, backgroundColor: '#28a745' }] },
            options: { responsive: true, plugins: { legend: { display: false } } }
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
            data: { labels: reasonLabels, datasets: [{ data: reasonData, backgroundColor: ['#007bff','#dc3545','#ffc107','#28a745','#6c757d','#17a2b8'] }] },
            options: { responsive: true }
          });
        } else {
          reasonChart.data.labels = reasonLabels;
          reasonChart.data.datasets[0].data = reasonData;
          reasonChart.update();
        }
        // Open positions table
        let openHtml = '<table class="table table-sm table-striped"><thead><tr>';
        if (data.open_trades.length > 0) {
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
        let logHtml = '<table class="table table-sm table-striped"><thead><tr>';
        if (data.trade_log.length > 0) {
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
    setInterval(updateDashboard, 2000);
    updateDashboard();
    </script>
    </body>
    </html>
    '''
    return render_template_string(html)

@app.route('/api/state')
def api_state():
    return jsonify(state)

# To run: from Paper Trading.webapp import app; app.run(debug=True, port=5000) 