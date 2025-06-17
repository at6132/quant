import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
import uuid
import redis
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Order:
    id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    type: str  # 'market', 'limit', 'stop'
    quantity: float
    price: float
    timestamp: datetime
    status: str  # 'new', 'partially_filled', 'filled', 'cancelled', 'rejected'
    filled_quantity: float = 0
    average_fill_price: float = 0
    fees: float = 0
    margin_required: float = 0
    leverage: float = 1.0
    stop_price: Optional[float] = None
    limit_price: Optional[float] = None
    time_in_force: str = 'GTC'  # GTC, IOC, FOK
    reduce_only: bool = False
    post_only: bool = False

class OrderManager:
    def __init__(self, config_path: str):
        """Initialize order manager with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Initialize Redis connection
        self.redis = redis.Redis(
            host='localhost',
            port=6379,
            db=0,
            decode_responses=True
        )
        
        # Order storage keys
        self.orders_key = 'trading:orders'
        self.positions_key = 'trading:positions'
        self.margin_key = 'trading:margin'
        
        # Load configuration
        self.margin_config = self.config['trading']['margin']
        self.fee_config = self.config['trading']['fees']
        self.slippage_config = self.config['trading']['slippage']
        
        # Initialize state
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Dict] = {}
        self.margin_account: Dict = {
            'equity': self.config['trading']['paper_trading']['initial_capital'],
            'used_margin': 0,
            'free_margin': self.config['trading']['paper_trading']['initial_capital'],
            'margin_level': 100,
            'unrealized_pnl': 0,
            'realized_pnl': 0
        }
        
    def create_order(self, 
                    symbol: str,
                    side: str,
                    quantity: float,
                    order_type: str = 'market',
                    price: Optional[float] = None,
                    stop_price: Optional[float] = None,
                    leverage: float = 1.0,
                    reduce_only: bool = False,
                    post_only: bool = False) -> Order:
        """Create a new order with realistic margin requirements"""
        # Generate order ID
        order_id = str(uuid.uuid4())
        
        # Calculate margin required
        margin_required = self._calculate_margin_required(
            symbol=symbol,
            quantity=quantity,
            price=price or self._get_current_price(symbol),
            leverage=leverage
        )
        
        # Check margin availability
        if not self._check_margin_availability(margin_required):
            raise ValueError(f"Insufficient margin available. Required: {margin_required:.2f}")
            
        # Create order
        order = Order(
            id=order_id,
            symbol=symbol,
            side=side,
            type=order_type,
            quantity=quantity,
            price=price or self._get_current_price(symbol),
            timestamp=datetime.now(),
            status='new',
            margin_required=margin_required,
            leverage=leverage,
            stop_price=stop_price,
            limit_price=price,
            reduce_only=reduce_only,
            post_only=post_only
        )
        
        # Store order
        self.orders[order_id] = order
        self._store_order(order)
        
        # Update margin account
        self._update_margin_account(order)
        
        return order
        
    def process_order(self, order: Order, current_price: float) -> Tuple[Order, float]:
        """Process an order with realistic fills and slippage"""
        if order.status not in ['new', 'partially_filled']:
            return order, 0
            
        # Calculate fill price with slippage
        fill_price = self._calculate_fill_price(
            order=order,
            current_price=current_price
        )
        
        # Calculate fill quantity
        fill_quantity = self._calculate_fill_quantity(
            order=order,
            current_price=current_price
        )
        
        if fill_quantity <= 0:
            return order, 0
            
        # Calculate fees
        fees = self._calculate_fees(
            quantity=fill_quantity,
            price=fill_price,
            side=order.side
        )
        
        # Update order
        order.filled_quantity += fill_quantity
        order.fees += fees
        
        # Calculate average fill price
        if order.filled_quantity > 0:
            order.average_fill_price = (
                (order.average_fill_price * (order.filled_quantity - fill_quantity) +
                 fill_price * fill_quantity) / order.filled_quantity
            )
            
        # Update order status
        if order.filled_quantity >= order.quantity:
            order.status = 'filled'
        else:
            order.status = 'partially_filled'
            
        # Update position
        self._update_position(order, fill_quantity, fill_price)
        
        # Update margin account
        self._update_margin_account(order)
        
        # Store updated order
        self._store_order(order)
        
        return order, fill_quantity
        
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order"""
        if order_id not in self.orders:
            return False
            
        order = self.orders[order_id]
        if order.status not in ['new', 'partially_filled']:
            return False
            
        # Update order status
        order.status = 'cancelled'
        
        # Release margin
        self._release_margin(order)
        
        # Store updated order
        self._store_order(order)
        
        return True
        
    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get current position for a symbol"""
        return self.positions.get(symbol)
        
    def get_margin_account(self) -> Dict:
        """Get current margin account status"""
        return self.margin_account
        
    def _calculate_margin_required(self, 
                                 symbol: str,
                                 quantity: float,
                                 price: float,
                                 leverage: float) -> float:
        """Calculate margin required for an order"""
        # Get position value
        position_value = quantity * price
        
        # Calculate initial margin
        initial_margin = position_value / leverage
        
        # Add maintenance margin
        maintenance_margin = position_value * self.margin_config['maintenance_margin_rate']
        
        return initial_margin + maintenance_margin
        
    def _check_margin_availability(self, margin_required: float) -> bool:
        """Check if sufficient margin is available"""
        return self.margin_account['free_margin'] >= margin_required
        
    def _calculate_fill_price(self, order: Order, current_price: float) -> float:
        """Calculate fill price with slippage"""
        if order.type == 'market':
            # Add random slippage
            slippage = np.random.normal(
                self.slippage_config['mean'],
                self.slippage_config['std']
            )
            return current_price * (1 + slippage if order.side == 'buy' else 1 - slippage)
        elif order.type == 'limit':
            if order.side == 'buy':
                return min(order.limit_price, current_price)
            else:
                return max(order.limit_price, current_price)
        elif order.type == 'stop':
            if order.side == 'buy':
                return max(order.stop_price, current_price)
            else:
                return min(order.stop_price, current_price)
        return current_price
        
    def _calculate_fill_quantity(self, order: Order, current_price: float) -> float:
        """Calculate fill quantity with partial fills"""
        if order.type == 'market':
            # Market orders are fully filled
            return order.quantity - order.filled_quantity
        elif order.type == 'limit':
            # Limit orders may be partially filled
            if order.side == 'buy' and current_price <= order.limit_price:
                return order.quantity - order.filled_quantity
            elif order.side == 'sell' and current_price >= order.limit_price:
                return order.quantity - order.filled_quantity
        elif order.type == 'stop':
            # Stop orders are fully filled when triggered
            if order.side == 'buy' and current_price >= order.stop_price:
                return order.quantity - order.filled_quantity
            elif order.side == 'sell' and current_price <= order.stop_price:
                return order.quantity - order.filled_quantity
        return 0
        
    def _calculate_fees(self, quantity: float, price: float, side: str) -> float:
        """Calculate trading fees"""
        # Get fee rate based on side
        fee_rate = (self.fee_config['maker_rate'] if side == 'buy' 
                   else self.fee_config['taker_rate'])
        
        # Calculate fee
        return quantity * price * fee_rate
        
    def _update_position(self, order: Order, fill_quantity: float, fill_price: float) -> None:
        """Update position after fill"""
        symbol = order.symbol
        
        if symbol not in self.positions:
            self.positions[symbol] = {
                'quantity': 0,
                'average_price': 0,
                'unrealized_pnl': 0,
                'realized_pnl': 0,
                'margin_used': 0
            }
            
        position = self.positions[symbol]
        
        if order.side == 'buy':
            # Update long position
            new_quantity = position['quantity'] + fill_quantity
            position['average_price'] = (
                (position['average_price'] * position['quantity'] +
                 fill_price * fill_quantity) / new_quantity
            )
            position['quantity'] = new_quantity
        else:
            # Update short position
            new_quantity = position['quantity'] - fill_quantity
            if new_quantity == 0:
                # Position closed
                position['realized_pnl'] += (
                    (position['average_price'] - fill_price) * fill_quantity
                )
                position['quantity'] = 0
                position['average_price'] = 0
            else:
                position['quantity'] = new_quantity
                
        # Update margin used
        position['margin_used'] = self._calculate_margin_required(
            symbol=symbol,
            quantity=abs(position['quantity']),
            price=fill_price,
            leverage=order.leverage
        )
        
        # Store position
        self._store_position(symbol, position)
        
    def _update_margin_account(self, order: Order) -> None:
        """Update margin account after order"""
        # Update used margin
        self.margin_account['used_margin'] = sum(
            p['margin_used'] for p in self.positions.values()
        )
        
        # Update free margin
        self.margin_account['free_margin'] = (
            self.margin_account['equity'] - self.margin_account['used_margin']
        )
        
        # Update margin level
        if self.margin_account['used_margin'] > 0:
            self.margin_account['margin_level'] = (
                self.margin_account['equity'] / self.margin_account['used_margin'] * 100
            )
        else:
            self.margin_account['margin_level'] = 100
            
        # Store margin account
        self._store_margin_account()
        
    def _release_margin(self, order: Order) -> None:
        """Release margin for cancelled order"""
        if order.status == 'new':
            self.margin_account['used_margin'] -= order.margin_required
            self.margin_account['free_margin'] += order.margin_required
            self._store_margin_account()
            
    def _get_current_price(self, symbol: str) -> float:
        """Get current price from Redis"""
        price_key = f"price:{symbol}"
        price = self.redis.get(price_key)
        return float(price) if price else 0.0
        
    def _store_order(self, order: Order) -> None:
        """Store order in Redis"""
        order_key = f"{self.orders_key}:{order.id}"
        self.redis.setex(
            order_key,
            timedelta(days=7),
            json.dumps(order.__dict__)
        )
        self.redis.lpush(self.orders_key, order_key)
        self.redis.ltrim(self.orders_key, 0, 999)
        
    def _store_position(self, symbol: str, position: Dict) -> None:
        """Store position in Redis"""
        position_key = f"{self.positions_key}:{symbol}"
        self.redis.setex(
            position_key,
            timedelta(days=1),
            json.dumps(position)
        )
        
    def _store_margin_account(self) -> None:
        """Store margin account in Redis"""
        self.redis.setex(
            self.margin_key,
            timedelta(days=1),
            json.dumps(self.margin_account)
        ) 