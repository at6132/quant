from flask import Flask, render_template_string, jsonify
import threading

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

@app.route('/')
def dashboard():
    html = '''
    <h1>Paper Trading Dashboard</h1>
    <p><b>Live BTC Price:</b> ${{ '{:,.2f}'.format(live_price) if live_price else 'N/A' }}</p>
    <p><b>Current Account Balance:</b> ${{ '{:,.2f}'.format(capital) }}</p>
    <h2>Open Position</h2>
    {% if open_trades %}
    <table border="1"><tr>{% for k in open_trades[0].keys() %}<th>{{k}}</th>{% endfor %}</tr>
    <tr>{% for v in open_trades[0].values() %}<td>{{v}}</td>{% endfor %}</tr></table>
    {% else %}<p>No open positions.</p>{% endif %}
    <h2>Trade Log (last 20)</h2>
    <pre>{{trade_log}}</pre>
    '''
    return render_template_string(
        html,
        capital=state['capital'],
        open_trades=state['open_trades'],
        trade_log=state['trade_log'][-20:],
        live_price=state['live_price']
    )

@app.route('/api/state')
def api_state():
    return jsonify(state)

# To run: from Paper Trading.webapp import app; app.run(debug=True, port=5000) 