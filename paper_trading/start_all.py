import threading
import time
import os
from webapp import app

webapp_thread = None
trader_thread = None

def run_webapp():
    app.run(debug=True, port=5000, use_reloader=False)

def run_trader():
    os.system('python paper_trader.py')

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
        if webapp_thread:
            pass
        if trader_thread:
            trader_thread.join(timeout=2)
        print("Shutdown complete.")
        os._exit(0) 