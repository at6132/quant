#!/usr/bin/env python3
"""
Simplified paper trading system startup script
Focuses on getting the 15-second candle loop running error-free
"""

import logging
import yaml
import signal
import sys
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import components
from paper_trading.trading_loop import TradingLoop

class SimpleTradingSystem:
    def __init__(self, config_path: str):
        """Initialize trading system with configuration"""
        self.config_path = config_path
        self.running = False
        
        # Load configuration
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise
            
        # Initialize trading loop (this includes data processing and 15s loop)
        try:
            self.trading_loop = TradingLoop(self.config)
            logger.info("Trading loop initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing trading loop: {str(e)}")
            raise
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        
    def start(self):
        """Start the trading system"""
        if self.running:
            logger.warning("Trading system is already running")
            return
            
        try:
            self.running = True
            logger.info("Starting 15-second candle trading loop...")
            
            # Start the trading loop (includes 15s processing)
            self.trading_loop.start()
            
            logger.info("Trading system started successfully - running 15s candle loop")
            logger.info("Trading symbols: %s", self.config['trading']['symbols'])
            logger.info("Timeframes: %s", self.config['trading']['timeframes'])
            logger.info("Press Ctrl+C to stop...")
            
            # Keep the main thread alive
            while self.running:
                try:
                    import time
                    time.sleep(1)
                except KeyboardInterrupt:
                    break
                    
        except Exception as e:
            logger.error(f"Error starting trading system: {str(e)}")
            self.stop()
            raise
            
    def stop(self):
        """Stop the trading system"""
        if not self.running:
            return
            
        logger.info("Stopping trading system...")
        self.running = False
        
        try:
            self.trading_loop.stop()
            logger.info("Trading system stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping trading system: {str(e)}")
            
    def handle_shutdown(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.stop()
        sys.exit(0)

def main():
    """Main entry point"""
    try:
        system = SimpleTradingSystem('paper_trading_config.yaml')
        system.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
        system.stop()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)
        
if __name__ == "__main__":
    main()