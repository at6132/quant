import redis
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clear_redis_data():
    """Clear all Redis data for market data and processed data"""
    try:
        # Connect to Redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        
        # Clear market data for all timeframes
        timeframes = ['15s', '1m', '5m', '15m', '1h', '4h']
        for tf in timeframes:
            r.delete(f'market_data:{tf}')
            r.delete(f'processed_data:{tf}')
            logger.info(f"Cleared data for timeframe: {tf}")
            
        # Clear raw market data
        r.delete('market_data:raw')
        logger.info("Cleared raw market data")
        
        logger.info("Successfully cleared all Redis data")
        
    except Exception as e:
        logger.error(f"Error clearing Redis data: {str(e)}")
        raise

if __name__ == "__main__":
    clear_redis_data() 