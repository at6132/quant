import threading
import time
import os
from webapp import app, update_state, set_live_price
from paper_trader import PaperTrader

webapp_thread = None
trader_thread = None
trader_instance = None

def run_webapp():
    app.run(debug=True, port=5000, use_reloader=False)

def run_trader():
    global trader_instance
    trader_instance = PaperTrader(
        model_path="Algorithm1/artefacts/lgbm_model.pkl",
        initial_capital=1_000_000,
        webapp_callback=update_state
    )
    # Patch the trader to update live price in webapp
    orig_on_new_data = trader_instance.on_new_data
    def patched_on_new_data(features_df):
        if features_df is not None and 'close' in features_df:
            set_live_price(features_df['close'].iloc[-1])
        orig_on_new_data(features_df)
    trader_instance.on_new_data = patched_on_new_data
    trader_instance.run()

if __name__ == "__main__":
    try:
        webapp_thread = threading.Thread(target=run_webapp)
        webapp_thread.daemon = True
        webapp_thread.start()
        time.sleep(2)
        trader_thread = threading.Thread(target=run_trader)
        trader_thread.start()
        while webapp_thread.is_alive() and trader_thread.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        if trader_instance:
            trader_instance.stop()
        if webapp_thread:
            pass
        if trader_thread:
            trader_thread.join(timeout=2)
        print("Shutdown complete.")
        os._exit(0) 