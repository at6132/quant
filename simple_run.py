#!/usr/bin/env python3
from paper_trading.webapp import WebApp
import yaml

with open('paper_trading_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

webapp = WebApp(config)
print('WebApp initialized, starting...')
try:
    webapp.start()
except KeyboardInterrupt:
    print("Stopping webapp...")
    webapp.stop()