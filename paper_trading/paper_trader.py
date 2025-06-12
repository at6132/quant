import os
import time
import pickle
import krakenex
from webapp import update_state, set_live_price
from data_processor import DataProcessor
from account import Account
from risk_manager import RiskManager

# Action mapping
ACTIONS = {
    0: 'HOLD',
    1: 'OPEN_LONG',
    2: 'CLOSE_LONG',
    3: 'OPEN_SHORT',
    4: 'CLOSE_SHORT',
    5: 'ADD_LONG',
    6: 'ADD_SHORT'
}

def main():
    # Load latest model
    model_path = 'artefacts/lgbm_model.pkl'
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Initialize account and risk manager
    account = Account(initial_capital=1_000_000)
    risk_manager = RiskManager(account)
    
    # Initialize data processor
    processor = DataProcessor()
    
    # Initialize WebSocket connection
    ws = krakenex.WebSocket()
    
    def on_message(msg):
        try:
            if isinstance(msg, dict) and 'price' in msg:
                price = float(msg['price'])
                print(f"\nReceived price: ${price:,.2f}")
                
                # Update web app state with live price
                set_live_price(price)
                
                # Process data and get features
                features = processor.process_live_data(price)
                if features is not None:
                    # Get model prediction
                    pred = model.predict([features])[0]
                    proba = model.predict_proba([features])[0]
                    
                    # Get action and probability
                    action = ACTIONS[pred]
                    action_prob = proba[pred]
                    
                    print(f"\nModel prediction: {action} (confidence: {action_prob:.2%})")
                    
                    # Execute trade with risk management
                    trade_result = risk_manager.execute_trade(action, action_prob, price)
                    if trade_result:
                        print(f"Trade executed: {trade_result}")
                        # Update web app state with new trade
                        update_state(account, trade_result)
                    
        except Exception as e:
            print(f"Error processing message: {e}")
    
    def on_error(error):
        print(f"WebSocket error: {error}")
    
    def on_close():
        print("WebSocket connection closed")
    
    def on_open():
        print("WebSocket connection established")
        # Subscribe to BTC/USD ticker
        ws.subscribe(['BTC/USD'], 'ticker')
    
    # Set up WebSocket callbacks
    ws.on_message = on_message
    ws.on_error = on_error
    ws.on_close = on_close
    ws.on_open = on_open
    
    # Start WebSocket connection
    ws.connect()
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        ws.close()
        print("WebSocket connection closed")
        print("Goodbye!")

if __name__ == "__main__":
    main() 