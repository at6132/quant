import logging
from datetime import datetime
from typing import Dict, List, Union, Optional
import yaml
import redis
import json
import uuid

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Order:
    def __init__(self, symbol: str, side: str, quantity: float, price: float):
        self.id = str(uuid.uuid4())
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.price = price
        self.timestamp = datetime.now().isoformat()
        self.status = "NEW"
        self.filled_quantity = 0
        self.average_fill_price = 0
        self.fees = 0
        self.margin_required = 0
        self.leverage = 1
        self.stop_price = None
        self.limit_price = None
        self.time_in_force = "GTC"
        self.reduce_only = False
        self.post_only = False

class OrderManager:
    def __init__(self, config: Union[str, Dict] = "paper_trading_config.yaml"):
        """Initialize order manager with configuration
        
        Args:
            config: Either a path to the config file or a config dictionary
        """
        if isinstance(config, str):
            with open(config, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config
            
        # Load settings
        self.trading_settings = self.config.get('trading', {})
        self.account_settings = self.config.get('account', {})
        
        # Initialize order settings with defaults
        self.min_order_size = self.trading_settings.get('min_order_size', 0.001)
        self.max_order_size = self.trading_settings.get('max_order_size', 1000.0)
        self.max_slippage = self.trading_settings.get('max_slippage', 0.01)  # 1%
        
        # Initialize state
        self.orders = {}
        self.positions = {}  # symbol -> {'side': 'long'/'short', 'quantity': float, 'entry_price': float}
        self.trade_history = []  # List of round-trip trades with realized PnL
        self.running = False
        
        # Initialize Redis connection
        if 'monitoring' not in self.config or 'redis' not in self.config['monitoring']:
            raise ValueError("Missing Redis configuration in 'monitoring' section")
            
        redis_config = self.config['monitoring']['redis']
        self.redis = redis.Redis(
            host=redis_config['host'],
            port=redis_config['port'],
            db=redis_config['db']
        )
        
        # Set up Redis keys
        self.orders_key = "orders"
        self.positions_key = "positions"
        self.margin_account_key = "margin_account"
        
        logger.info("Order manager initialized")
        
    def start(self):
        """Start the order manager"""
        if self.running:
            logger.warning("Order manager is already running")
            return
            
        try:
            self.running = True
            logger.info("Order manager started")
        except Exception as e:
            logger.error("Error starting order manager: %s", str(e))
            self.running = False
            raise
            
    def stop(self):
        """Stop the order manager"""
        if not self.running:
            logger.warning("Order manager is not running")
            return
            
        try:
            self.running = False
            logger.info("Order manager stopped")
        except Exception as e:
            logger.error("Error stopping order manager: %s", str(e))
            raise
            
    def place_order(self, symbol: str, side: str, size: float, price: float) -> Optional[Dict]:
        """Place a new order and record a trade only when a position is closed or reduced, with correct realized PnL."""
        try:
            order_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            realized_pnl = 0.0
            trade_recorded = False
            trade = None

            # Get current position
            pos = self.positions.get(symbol, {'side': None, 'quantity': 0.0, 'entry_price': 0.0})
            pos_side = pos['side']
            pos_qty = pos['quantity']
            pos_entry = pos['entry_price']

            if pos_side is None or pos_qty == 0.0:
                # No open position, this is an entry
                new_side = 'long' if side == 'buy' else 'short'
                self.positions[symbol] = {
                    'side': new_side,
                    'quantity': size,
                    'entry_price': price
                }
            elif (pos_side == 'long' and side == 'buy') or (pos_side == 'short' and side == 'sell'):
                # Adding to existing position
                total_qty = pos_qty + size
                avg_entry = (pos_entry * pos_qty + price * size) / total_qty
                self.positions[symbol]['quantity'] = total_qty
                self.positions[symbol]['entry_price'] = avg_entry
            else:
                # Closing or reducing position
                closing_qty = min(size, pos_qty)
                if pos_side == 'long':
                    realized_pnl = (price - pos_entry) * closing_qty
                else:
                    realized_pnl = (pos_entry - price) * closing_qty
                
                # Record the round-trip trade
                trade = {
                    'id': order_id,
                    'symbol': symbol,
                    'entry_side': pos_side,
                    'entry_price': pos_entry,
                    'exit_side': side,
                    'exit_price': price,
                    'quantity': closing_qty,
                    'realized_pnl': realized_pnl,
                    'timestamp': timestamp
                }
                self.trade_history.append(trade)
                trade_recorded = True
                
                # Update or close the position
                remaining_qty = pos_qty - closing_qty
                if remaining_qty > 0:
                    self.positions[symbol]['quantity'] = remaining_qty
                    # entry_price remains the same
                else:
                    self.positions[symbol] = {'side': None, 'quantity': 0.0, 'entry_price': 0.0}
                
                # If order size > closing_qty, open new position in opposite direction
                open_qty = size - closing_qty
                if open_qty > 0:
                    new_side = 'long' if side == 'buy' else 'short'
                    self.positions[symbol] = {
                        'side': new_side,
                        'quantity': open_qty,
                        'entry_price': price
                    }
            # Store order (for completeness, but not used for trade history)
            order = {
                'id': order_id,
                'symbol': symbol,
                'side': side,
                'size': size,
                'price': price,
                'timestamp': timestamp,
                'status': 'filled',
                'realized_pnl': realized_pnl if trade_recorded else None
            }
            self.orders[order_id] = order
            logger.info("Placed %s order for %s: %.6f @ %.2f%s", side, symbol, size, price, f" (Trade PnL: {realized_pnl:.2f})" if trade_recorded else "")
            return order
        except Exception as e:
            logger.error("Error placing order: %s", str(e))
            return None
        
    def get_position(self, symbol: str) -> float:
        """Get current position for a symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current position size
        """
        return self.positions.get(symbol, 0.0)
        
    def get_positions(self) -> Dict[str, float]:
        """Get all current positions
        
        Returns:
            Dictionary of symbol to position size
        """
        return self.positions.copy()
        
    def get_orders(self) -> List[Dict]:
        """Get order history
        
        Returns:
            List of order dictionaries
        """
        return self.trade_history.copy()
        
    def create_order(self, symbol: str, side: str, quantity: float, price: float) -> Dict:
        """Create a new order"""
        try:
            # Validate order parameters
            if quantity < self.min_order_size:
                raise ValueError(f"Order size {quantity} below minimum {self.min_order_size}")
            if quantity > self.max_order_size:
                raise ValueError(f"Order size {quantity} above maximum {self.max_order_size}")
                
            # Generate order ID
            order_id = str(uuid.uuid4())
            
            # Create order
            order = {
                'id': order_id,
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': price,
                'status': 'new',
                'timestamp': datetime.now().isoformat(),
                'filled_quantity': 0.0,
                'filled_price': 0.0,
                'slippage': 0.0
            }
            
            # Store order
            self.orders[order_id] = order
            
            logger.info("Created order %s: %s %s %.4f @ %.2f", 
                       order_id, side, symbol, quantity, price)
            
            return order
            
        except Exception as e:
            logger.error("Error creating order: %s", str(e))
            raise
        
    def process_order(self, order_id: str, fill_price: float) -> Dict:
        """Process an order with slippage"""
        try:
            if order_id not in self.orders:
                raise ValueError(f"Order {order_id} not found")
                
            order = self.orders[order_id]
            
            # Calculate slippage
            slippage = abs(fill_price - order['price']) / order['price']
            if slippage > self.max_slippage:
                raise ValueError(f"Slippage {slippage:.2%} exceeds maximum {self.max_slippage:.2%}")
                
            # Update order
            order['status'] = 'filled'
            order['filled_quantity'] = order['quantity']
            order['filled_price'] = fill_price
            order['slippage'] = slippage
            order['fill_timestamp'] = datetime.now().isoformat()
            
            # Update position
            self._update_position(order)
            
            # Add to history
            self.trade_history.append(order)
            
            # Remove from active orders
            del self.orders[order_id]
            
            logger.info("Processed order %s: %s %s %.4f @ %.2f (slippage: %.2%%)",
                       order_id, order['side'], order['symbol'], 
                       order['filled_quantity'], order['filled_price'], slippage)
            
            return order
            
        except Exception as e:
            logger.error("Error processing order: %s", str(e))
            raise
        
    def _update_position(self, order: Dict):
        """Update position after order fill"""
        try:
            symbol = order['symbol']
            side = order['side']
            quantity = order['filled_quantity']
            price = order['filled_price']
            
            if symbol not in self.positions:
                self.positions[symbol] = {
                    'quantity': 0.0,
                    'average_price': 0.0,
                    'side': None
                }
                
            position = self.positions[symbol]
            
            if position['side'] is None:
                position['side'] = side
                position['quantity'] = quantity
                position['average_price'] = price
            elif position['side'] == side:
                # Add to position
                total_quantity = position['quantity'] + quantity
                total_value = (position['quantity'] * position['average_price'] + 
                             quantity * price)
                position['quantity'] = total_quantity
                position['average_price'] = total_value / total_quantity
            else:
                # Reduce position
                if quantity > position['quantity']:
                    raise ValueError("Cannot close more than position size")
                position['quantity'] -= quantity
                if position['quantity'] == 0:
                    position['side'] = None
                    position['average_price'] = 0.0
                    
            logger.info("Updated position for %s: %s %.4f @ %.2f",
                       symbol, position['side'], position['quantity'], 
                       position['average_price'])
            
        except Exception as e:
            logger.error("Error updating position: %s", str(e))
            raise
        
    def get_trade_history(self) -> List[Dict]:
        """Get round-trip trade history (entry/exit pairs with realized PnL)"""
        return self.trade_history.copy()
        
    def get_open_orders(self) -> List[Dict]:
        """Get open orders"""
        return [order for order in self.orders.values() 
                if order['status'] == 'new']
        
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order"""
        order_data = self.redis.hget(self.orders_key, order_id)
        if not order_data:
            logger.warning("Order %s not found", order_id)
            return False
            
        order = json.loads(order_data)
        if order['status'] != "NEW":
            logger.warning("Order %s cannot be cancelled: status is %s", 
                         order_id, order['status'])
            return False
            
        order['status'] = "CANCELLED"
        self.redis.hset(self.orders_key, order_id, json.dumps(order))
        
        logger.info("Order %s cancelled", order_id)
        return True
        
    def _calculate_margin_required(self, order: Order) -> float:
        """Calculate margin required for an order"""
        return order.quantity * order.price * self.config['account']['margin_requirement']
        
    def _check_margin_available(self, order: Order) -> bool:
        """Check if sufficient margin is available"""
        margin_account = json.loads(self.redis.get(self.margin_account_key) or '{}')
        available_margin = margin_account.get('available_margin', 0)
        return available_margin >= order.margin_required
        
    def _simulate_fill_price(self, order: Order) -> float:
        """Simulate order fill price with slippage"""
        slippage = self.config['trading']['max_slippage']
        if order.side == 'buy':
            return order.price * (1 + slippage)
        return order.price * (1 - slippage)
        
    def _calculate_fees(self, order: Order, fill_price: float, fill_quantity: float) -> float:
        """Calculate trading fees"""
        return fill_price * fill_quantity * self.config['trading']['commission']
        
    def _update_positions(self, order: Order, fill_price: float, fill_quantity: float):
        """Update positions after order fill"""
        position_key = f"{order.symbol}:{order.side}"
        position_data = self.redis.hget(self.positions_key, position_key)
        
        if position_data:
            position = json.loads(position_data)
            if position['side'] == order.side:
                # Increase position
                position['quantity'] += fill_quantity
                position['average_price'] = (
                    (position['average_price'] * position['quantity'] + 
                     fill_price * fill_quantity) / 
                    (position['quantity'] + fill_quantity)
                )
            else:
                # Reduce position
                position['quantity'] -= fill_quantity
                if position['quantity'] <= 0:
                    self.redis.hdel(self.positions_key, position_key)
                    return
        else:
            position = {
                'symbol': order.symbol,
                'side': order.side,
                'quantity': fill_quantity,
                'average_price': fill_price
            }
            
        self.redis.hset(self.positions_key, position_key, json.dumps(position))
        
    def _update_margin_account(self, order: Order, fill_price: float, 
                             fill_quantity: float, fees: float):
        """Update margin account after order fill"""
        margin_account = json.loads(self.redis.get(self.margin_account_key) or '{}')
        
        # Update available margin
        margin_account['available_margin'] -= order.margin_required
        
        # Update P&L
        if order.side == 'buy':
            margin_account['unrealized_pnl'] -= fill_price * fill_quantity
        else:
            margin_account['unrealized_pnl'] += fill_price * fill_quantity
            
        # Update fees
        margin_account['total_fees'] += fees
        
        self.redis.set(self.margin_account_key, json.dumps(margin_account)) 