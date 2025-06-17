#!/usr/bin/env python3
"""
Start analytics dashboard that connects to existing trading system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import yaml
import time
import redis
import json
from paper_trading.web_app import WebApp

def main():
    """Start analytics dashboard connected to existing trading system"""
    try:
        # Load config
        config_path = os.path.join(os.path.dirname(__file__), 'paper_trading_config.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print("Starting analytics dashboard...")
        print("Connecting to existing trading system data...")
        
        # Test connection to Redis (where trading system stores data)
        redis_config = config.get('monitoring', {}).get('redis', {})
        try:
            redis_client = redis.Redis(
                host=redis_config.get('host', 'localhost'),
                port=redis_config.get('port', 6379),
                db=redis_config.get('db', 0)
            )
            redis_client.ping()
            print("✓ Connected to trading system Redis")
            
            # Print initial data found in Redis
            print("\n=== Redis Data Check ===")
            keys = redis_client.keys('*')
            print(f"Found {len(keys)} keys in Redis:")
            
            for key in keys:
                key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                try:
                    # Check data type first
                    key_type = redis_client.type(key)
                    key_type_str = key_type.decode('utf-8') if isinstance(key_type, bytes) else key_type
                    
                    if key_type_str == 'string':
                        value = redis_client.get(key)
                        if value:
                            value_str = value.decode('utf-8') if isinstance(value, bytes) else str(value)
                            print(f"  {key_str} (string): {value_str}")
                        else:
                            print(f"  {key_str} (string): (empty)")
                    elif key_type_str == 'list':
                        length = redis_client.llen(key)
                        print(f"  {key_str} (list): {length} items")
                    elif key_type_str == 'hash':
                        length = redis_client.hlen(key)
                        print(f"  {key_str} (hash): {length} fields")
                    elif key_type_str == 'set':
                        length = redis_client.scard(key)
                        print(f"  {key_str} (set): {length} members")
                    else:
                        print(f"  {key_str} ({key_type_str}): unknown type")
                        
                except Exception as e:
                    print(f"  {key_str}: Error reading - {e}")
            
            print("=== End Redis Data ===\n")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not connect to Redis: {e}")
            print("Dashboard will show placeholder data")
        
        # Initialize web app
        web_app = WebApp(config)
        
        # Start web application
        print("Starting analytics dashboard...")
        web_app.start()
        
        print(f"Dashboard started at: http://localhost:{config.get('webapp', {}).get('port', 5000)}")
        print("This dashboard will:")
        print("- Connect to your running trading system data")
        print("- Show real-time positions, PnL, and metrics")
        print("- Display all trading analytics")
        print("- Read from the same data sources as your trading system")
        print("Press Ctrl+C to stop...")
        
        # Keep running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping dashboard...")
            web_app.stop()
            print("Dashboard stopped successfully")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 