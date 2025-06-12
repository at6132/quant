import os
import json

class Account:
    def __init__(self, initial_capital, fee_rate=0.001, slippage=0.0005):
        self.capital = initial_capital
        self.open_positions = []  # List of dicts
        self.trade_log = []
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.analytics_dir = os.path.join(os.path.dirname(__file__), '..', 'analytics')
        os.makedirs(self.analytics_dir, exist_ok=True)
        self.analytics_file = os.path.join(self.analytics_dir, 'trades.jsonl')

    def open_trade(self, price, size, direction, timestamp, tp_pct=0.01, sl_pct=0.005):
        if size > self.capital:
            return False  # Not enough capital
        entry_price = price * (1 + self.slippage if direction == 'long' else 1 - self.slippage)
        tp = entry_price * (1 + tp_pct) if direction == 'long' else entry_price * (1 - tp_pct)
        sl = entry_price * (1 - sl_pct) if direction == 'long' else entry_price * (1 + sl_pct)
        pos = {
            'entry_price': entry_price,
            'size': size,
            'direction': direction,
            'open_time': timestamp,
            'tp': tp,
            'sl': sl
        }
        self.open_positions.append(pos)
        self.capital -= size
        return True

    def close_trade(self, pos, exit_price, timestamp, reason='manual'):
        # Apply slippage to exit
        exit_price_adj = exit_price * (1 - self.slippage if pos['direction'] == 'long' else 1 + self.slippage)
        gross_pnl = (exit_price_adj - pos['entry_price']) * pos['size'] if pos['direction'] == 'long' else (pos['entry_price'] - exit_price_adj) * pos['size']
        fees = (pos['entry_price'] + exit_price_adj) * pos['size'] * self.fee_rate
        net_pnl = gross_pnl - fees
        self.capital += pos['size'] + net_pnl
        trade = {
            **pos,
            'exit_price': exit_price_adj,
            'exit_time': timestamp,
            'gross_pnl': gross_pnl,
            'fees': fees,
            'net_pnl': net_pnl,
            'capital': self.capital,
            'reason': reason
        }
        self.trade_log.append(trade)
        self.open_positions.remove(pos)
        # Save trade to analytics
        with open(self.analytics_file, 'a') as f:
            f.write(json.dumps(trade) + '\n')
        return trade

    def check_tp_sl(self, price, timestamp):
        closed = []
        for pos in self.open_positions[:]:
            hit_tp = (pos['direction'] == 'long' and price >= pos['tp']) or (pos['direction'] == 'short' and price <= pos['tp'])
            hit_sl = (pos['direction'] == 'long' and price <= pos['sl']) or (pos['direction'] == 'short' and price >= pos['sl'])
            if hit_tp:
                closed.append(self.close_trade(pos, pos['tp'], timestamp, reason='take-profit'))
            elif hit_sl:
                closed.append(self.close_trade(pos, pos['sl'], timestamp, reason='stop-loss'))
        return closed

    def get_open_trades(self):
        return self.open_positions

    def get_trade_log(self):
        return self.trade_log

    def get_capital(self):
        return self.capital 