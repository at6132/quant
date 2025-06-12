import threading
import time
from webapp import app, update_state
from paper_trader import PaperTrader

def run_webapp():
    app.run(debug=True, port=5000, use_reloader=False)

def run_trader():
    # Initialize paper trader with pre-trained model and live data
    trader = PaperTrader(
        model_path="Algorithm1/artefacts/lgbm_model.pkl",  # Correct path to your pre-trained model
        initial_capital=1_000_000,
        webapp_callback=update_state
    )
    trader.run()

if __name__ == "__main__":
    # Start web app in a separate thread
    t1 = threading.Thread(target=run_webapp)
    t1.start()
    
    # Give the web app a moment to start
    time.sleep(2)
    
    # Start the paper trader
    t2 = threading.Thread(target=run_trader)
    t2.start()
    
    # Wait for both threads
    t1.join()
    t2.join() 