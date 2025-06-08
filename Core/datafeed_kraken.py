import websocket
import json
import time
import threading
from datetime import datetime
import logging
import pandas as pd
import numpy as np
from Core.indicators.breaker_signals import LiveBreakerSignals
from Core.indicators.session_levels import LiveSessionLevels
from Core.indicators.liquidity_swings import LiveLiquiditySwings
from Core.indicators.tr_reality_core import LiveTRReality
from Core.indicators.smc_core import LiveSMC
from Core.indicators.pvsra_vs import pvsra_vs as pvsra_vs_func
from Core.indicators.IT_Foundation import process as it_foundation_process
from Core.indicators.sessions import build_session_table
from Core.indicators.ict_sm_trades import run as ict_sm_trades_run

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KrakenDataFeed:
    def __init__(self, symbol="XBT/USD"):
        self.symbol = symbol
        self.ws = None
        self.last_data = None
        self.running = False
        self.callback = None
        
        # Initialize all indicators
        self.breaker = LiveBreakerSignals()
        self.sessions = LiveSessionLevels()
        self.swings = LiveLiquiditySwings()
        self.tr = LiveTRReality()
        self.smc = LiveSMC()
        
        # Data buffers for indicators that need historical data
        self.data_buffer = []
        self.max_buffer_size = 2000  # Adjust based on your needs
        
        # Add callback to handle indicator results
        self.breaker.add_callback(self._handle_indicator_results)
        self.sessions.add_callback(self._handle_indicator_results)
        self.swings.add_callback(self._handle_indicator_results)
        self.tr.add_callback(self._handle_indicator_results)
        self.smc.add_callback(self._handle_indicator_results)

    def _handle_indicator_results(self, results):
        """Handle results from any indicator"""
        if self.callback:
            self.callback(results)

    def _process_historical_indicators(self, df):
        """Process indicators that need historical data"""
        results = {}
        
        # PVSRA VS
        try:
            pvsra_results = pvsra_vs_func(df)
            results['pvsra'] = {
                'vec_color': int(pvsra_results['vec_color'].iloc[-1]),
                'gr_pattern': bool(pvsra_results['gr_pattern'].iloc[-1])
            }
        except Exception as e:
            logger.error(f"PVSRA processing error: {e}")
            results['pvsra'] = {'vec_color': 0, 'gr_pattern': False}
        
        # IT Foundation
        it_results = it_foundation_process({'15s': df})
        results['it_foundation'] = it_results.iloc[-1].to_dict()
        
        # ICT SM Trades
        ict_results = ict_sm_trades_run(df)
        results['ict_sm_trades'] = ict_results.iloc[-1].to_dict()
        
        # Sessions
        session_results = build_session_table(df)
        results['sessions'] = session_results.iloc[-1].to_dict()
        
        return results

    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            if isinstance(data, list) and len(data) > 1:
                # Extract relevant data from the ticker message
                ticker_data = data[1]
                if isinstance(ticker_data, dict):
                    self.last_data = {
                        'timestamp': datetime.now().timestamp(),
                        'symbol': self.symbol,
                        'price': float(ticker_data.get('c', [0])[0]),  # Current price
                        'volume': float(ticker_data.get('v', [0])[0]),  # 24h volume
                        'high': float(ticker_data.get('h', [0])[0]),   # 24h high
                        'low': float(ticker_data.get('l', [0])[0]),    # 24h low
                        'vwap': float(ticker_data.get('p', [0])[0])    # VWAP
                    }
                    
                    # Add to buffer
                    self.data_buffer.append(self.last_data)
                    if len(self.data_buffer) > self.max_buffer_size:
                        self.data_buffer.pop(0)
                    
                    # Convert buffer to DataFrame
                    df = pd.DataFrame(self.data_buffer)
                    df.set_index('timestamp', inplace=True)
                    # Ensure OHLCV format
                    df = df.rename(columns={
                        'price': 'close',
                        'high': 'high',
                        'low': 'low',
                        'volume': 'volume'
                    })
                    df['open'] = df['close'].shift(1).fillna(df['close'])
                    
                    # Process data through all indicators
                    self.breaker.process_live_data(self.last_data)
                    self.sessions.process_live_data(self.last_data)
                    self.swings.process_live_data(self.last_data)
                    self.tr.process_live_data(self.last_data)
                    self.smc.process_live_data(self.last_data)
                    
                    # Process historical indicators
                    hist_results = self._process_historical_indicators(df)
                    if self.callback:
                        self.callback({
                            'timestamp': self.last_data['timestamp'],
                            'price': self.last_data['price'],
                            **hist_results
                        })
                    
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    def on_error(self, ws, error):
        logger.error(f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        logger.info("WebSocket connection closed")
        if self.running:
            self.reconnect()

    def on_open(self, ws):
        logger.info("WebSocket connection established")
        # Subscribe to ticker
        subscribe_message = {
            "event": "subscribe",
            "pair": [self.symbol],
            "subscription": {
                "name": "ticker"
            }
        }
        ws.send(json.dumps(subscribe_message))

    def reconnect(self):
        logger.info("Attempting to reconnect...")
        time.sleep(5)  # Wait 5 seconds before reconnecting
        self.start()

    def start(self):
        self.running = True
        websocket.enableTrace(True)
        self.ws = websocket.WebSocketApp(
            "wss://ws.kraken.com",
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open
        )
        
        # Start WebSocket connection in a separate thread
        self.ws_thread = threading.Thread(target=self.ws.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()

    def stop(self):
        self.running = False
        if self.ws:
            self.ws.close()

    def set_callback(self, callback):
        self.callback = callback

def main():
    # Example usage
    def data_callback(results):
        logger.info(f"Received indicator results: {results}")

    feed = KrakenDataFeed()
    feed.set_callback(data_callback)
    feed.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping data feed...")
        feed.stop()

if __name__ == "__main__":
    main()
