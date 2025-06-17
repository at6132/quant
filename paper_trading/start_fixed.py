#!/usr/bin/env python3
"""
Fixed Paper Trading System Startup Script
This script runs the paper trading system with corrected position size calculations.
"""

import sys
import os
import time
import logging
import yaml
from datetime import datetime

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paper_trading.data_processor import DataProcessor
from paper_trading.risk_manager import RiskManager
from paper_trading.order_manager import OrderManager
from paper_trading.trading_loop import TradingLoop

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/paper_trading_fixed.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main function to start the fixed paper trading system"""
    
    try:
        logger.info("=== Starting Fixed Paper Trading System ===")
        
        # Load configuration
        config_path = 'paper_trading/paper_trading_config.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info("Configuration loaded successfully")
        
        # Initialize components
        logger.info("Initializing components...")
        
        data_processor = DataProcessor(config)
        risk_manager = RiskManager(config)
        order_manager = OrderManager(config)
        
        # Initialize trading loop
        trading_loop = TradingLoop(config, data_processor, risk_manager, order_manager)
        
        logger.info("All components initialized successfully")
        
        # Start components
        logger.info("Starting components...")
        
        data_processor.start()
        risk_manager.start()
        order_manager.start()
        
        logger.info("All components started successfully")
        
        # Start trading loop (this will run the main trading logic)
        logger.info("Starting trading loop with 15-second intervals...")
        logger.info("Press Ctrl+C to stop the system")
        
        trading_loop.start()
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise
    finally:
        # Cleanup
        try:
            if 'trading_loop' in locals():
                trading_loop.stop()
            if 'order_manager' in locals():
                order_manager.stop()
            if 'risk_manager' in locals():
                risk_manager.stop()
            if 'data_processor' in locals():
                data_processor.stop()
            logger.info("All components stopped successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

if __name__ == "__main__":
    main() 