import time
import joblib
from risk_module import RiskManager
from account import Account
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Core.datafeed_kraken import KrakenDataFeed
from utils import align_features
import threading
import pandas as pd

# Action mapping
ACTION_MAP = {
    0: 'hold',
    1: 'open_long',
    2: 'open_short',
    3: 'close_long',
    4: 'close_short',
    5: 'add_long',
    6: 'add_short'
}

class PaperTrader:
    def __init__(self, model_path, initial_capital, webapp_callback=None):
        self.model = joblib.load(model_path)
        self.account = Account(initial_capital)
        self.risk_manager = RiskManager()
        self.data_feed = KrakenDataFeed(symbol="XBT/USD")
        self.data_feed.set_callback(self.on_new_data)
        self.webapp_callback = webapp_callback
        self.last_features = None
        self.last_price = None
        self.last_timestamp = None
        self.holding_period = 15 * 4  # e.g. 1 minute (4x 15s bars)
        self.max_holding_period = 15 * 20  # e.g. 5 minutes
        self.running = False
        self.tp_pct = 0.01
        self.sl_pct = 0.005
        self.position_holding = {}  # pos_id: bars held
        self.pos_counter = 0

    def on_new_data(self, features_df):
        if features_df is None or not isinstance(features_df, pd.DataFrame) or features_df.empty:
            return
        X = align_features(features_df, self.model.feature_name())
        # Get action probabilities
        if hasattr(self.model, 'predict_proba'):
            proba_vec = self.model.predict_proba(X)[-1]
        else:
            proba_vec = self.model.predict(X)[-1]
        action_idx = int(proba_vec.argmax())
        action = ACTION_MAP.get(action_idx, 'hold')
        action_proba = proba_vec[action_idx]
        price = features_df['close'].iloc[-1]
        timestamp = features_df['timestamp'].iloc[-1] if 'timestamp' in features_df else time.time()
        trade_results = []
        print(f"[SCAN] Action: {action} (prob={action_proba:.2f}) at price {price}")
        # Check TP/SL for all open positions
        closed = self.account.check_tp_sl(price, timestamp)
        for trade in closed:
            print(f"[EXEC] Closed position: {trade}")
            self.risk_manager.update_after_trade(trade['net_pnl'])
            trade_results.append(trade)
        # Update holding period for each open position
        for pos in self.account.open_positions:
            pos_id = id(pos)
            self.position_holding[pos_id] = self.position_holding.get(pos_id, 0) + 1
        # Close positions that exceed max holding period
        for pos in self.account.open_positions[:]:
            pos_id = id(pos)
            if self.position_holding.get(pos_id, 0) >= self.max_holding_period:
                trade = self.account.close_trade(pos, price, timestamp, reason='max-hold')
                print(f"[EXEC] Closed position (max-hold): {trade}")
                self.risk_manager.update_after_trade(trade['net_pnl'])
                trade_results.append(trade)
                self.position_holding.pop(pos_id, None)
        # --- Action logic ---
        if action == 'open_long':
            size = self.risk_manager.get_position_size(action_proba, self.account)
            if size > 0 and self.account.capital >= size:
                opened = self.account.open_trade(price, size, 'long', timestamp, tp_pct=self.tp_pct, sl_pct=self.sl_pct)
                if opened:
                    pos = self.account.open_positions[-1]
                    self.position_holding[id(pos)] = 0
                    print(f"[EXEC] Opened LONG: {size} at {price}")
        elif action == 'open_short':
            size = self.risk_manager.get_position_size(action_proba, self.account)
            if size > 0 and self.account.capital >= size:
                opened = self.account.open_trade(price, size, 'short', timestamp, tp_pct=self.tp_pct, sl_pct=self.sl_pct)
                if opened:
                    pos = self.account.open_positions[-1]
                    self.position_holding[id(pos)] = 0
                    print(f"[EXEC] Opened SHORT: {size} at {price}")
        elif action == 'add_long':
            size = self.risk_manager.get_position_size(action_proba, self.account)
            if size > 0 and self.account.capital >= size:
                opened = self.account.open_trade(price, size, 'long', timestamp, tp_pct=self.tp_pct, sl_pct=self.sl_pct)
                if opened:
                    pos = self.account.open_positions[-1]
                    self.position_holding[id(pos)] = 0
                    print(f"[EXEC] Added LONG: {size} at {price}")
        elif action == 'add_short':
            size = self.risk_manager.get_position_size(action_proba, self.account)
            if size > 0 and self.account.capital >= size:
                opened = self.account.open_trade(price, size, 'short', timestamp, tp_pct=self.tp_pct, sl_pct=self.sl_pct)
                if opened:
                    pos = self.account.open_positions[-1]
                    self.position_holding[id(pos)] = 0
                    print(f"[EXEC] Added SHORT: {size} at {price}")
        elif action == 'close_long':
            for pos in self.account.open_positions[:]:
                if pos['direction'] == 'long':
                    trade = self.account.close_trade(pos, price, timestamp, reason='model-close')
                    print(f"[EXEC] Closed LONG: {trade}")
                    self.risk_manager.update_after_trade(trade['net_pnl'])
                    trade_results.append(trade)
                    self.position_holding.pop(id(pos), None)
        elif action == 'close_short':
            for pos in self.account.open_positions[:]:
                if pos['direction'] == 'short':
                    trade = self.account.close_trade(pos, price, timestamp, reason='model-close')
                    print(f"[EXEC] Closed SHORT: {trade}")
                    self.risk_manager.update_after_trade(trade['net_pnl'])
                    trade_results.append(trade)
                    self.position_holding.pop(id(pos), None)
        # No action for 'hold'
        if self.webapp_callback:
            self.webapp_callback(self.account, trade_results)

    def run(self):
        self.running = True
        self.data_feed.start()
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        self.running = False
        self.data_feed.stop()

if __name__ == "__main__":
    trader = PaperTrader("Algorithm1/artefacts/lgbm_model.pkl", initial_capital=1_000_000)
    trader.run() 