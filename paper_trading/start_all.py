import logging
logging.basicConfig(level=logging.DEBUG)
import threading
import time
import yaml
from pathlib import Path
import signal
import sys
import os
from datetime import datetime
from typing import Optional

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()

# Configure logging
os.makedirs(SCRIPT_DIR / 'logs', exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(SCRIPT_DIR / 'logs' / f'trading_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import components
from paper_trading.data_processor import DataProcessor
from Algorithm1.risk_manager import AdaptiveRiskManager
from paper_trading.monitoring import MonitoringSystem
from paper_trading.paper_trader import PaperTrader
from paper_trading.webapp import WebApp

class TradingPipeline:
    def __init__(self, config_path: str):
        """Initialize trading pipeline with configuration"""
        self.config_path = config_path
        self.running = False
        self.threads = []
        self.stop_event = threading.Event()
        
        # Load configuration
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise
            
        # Initialize components
        self.initialize_components()
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        
    def initialize_components(self):
        """Initialize all trading components"""
        try:
            # Initialize webapp (which includes trading loop)
            self.webapp = WebApp(self.config)
            
            # Initialize paper trader
            self.paper_trader = PaperTrader(self.config)
            
            # Initialize data processor
            self.data_processor = DataProcessor(self.config)
            
            # Initialize monitoring system
            self.monitoring = MonitoringSystem(self.config)
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            raise
            
    def start(self):
        """Start the trading pipeline"""
        if self.running:
            logger.warning("Trading pipeline is already running")
            return
            
        try:
            self.running = True
            self.stop_event.clear()
            
            # Start webapp (which includes trading loop)
            webapp_thread = threading.Thread(
                target=self.webapp.start,
                name="WebApp"
            )
            webapp_thread.daemon = True
            webapp_thread.start()
            self.threads.append(webapp_thread)
            
            # Start paper trader
            trader_thread = threading.Thread(
                target=self.paper_trader.start,
                name="PaperTrader"
            )
            trader_thread.daemon = True
            trader_thread.start()
            self.threads.append(trader_thread)
            
            # Start data processor
            processor_thread = threading.Thread(
                target=self.data_processor.start,
                name="DataProcessor"
            )
            processor_thread.daemon = True
            processor_thread.start()
            self.threads.append(processor_thread)
            
            # Start monitoring system
            monitoring_thread = threading.Thread(
                target=self.monitoring.start,
                name="MonitoringSystem"
            )
            monitoring_thread.daemon = True
            monitoring_thread.start()
            self.threads.append(monitoring_thread)
            
            logger.info("Trading pipeline started successfully")
            
            # Main loop
            while self.running and not self.stop_event.is_set():
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"Error starting trading pipeline: {str(e)}")
            self.stop()
            raise
            
    def stop(self):
        """Stop the trading pipeline"""
        if not self.running:
            return
            
        logger.info("Stopping trading pipeline...")
        self.running = False
        self.stop_event.set()
        
        # Stop all components
        try:
            self.webapp.stop()
            self.paper_trader.stop()
            self.data_processor.stop()
            self.monitoring.stop()
        except Exception as e:
            logger.error(f"Error stopping components: {str(e)}")
            
        # Wait for threads to finish
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=5.0)
                
        self.threads.clear()
        logger.info("Trading pipeline stopped")
        
    def handle_shutdown(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.stop()
        sys.exit(0)

def main():
    """Main entry point"""
    try:
        pipeline = TradingPipeline('paper_trading_config.yaml')
        pipeline.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
        pipeline.stop()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        if 'pipeline' in locals():
            pipeline.stop()
        sys.exit(1)
        
if __name__ == "__main__":
    main() 