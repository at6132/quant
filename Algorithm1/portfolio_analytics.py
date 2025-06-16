import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
import redis
import json
from scipy import stats
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PortfolioMetrics:
    total_value: float
    cash: float
    positions_value: float
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    daily_pnl: float
    daily_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    current_drawdown: float
    win_rate: float
    profit_factor: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    average_trade_duration: float
    position_count: int
    leverage: float
    margin_level: float
    beta: float
    alpha: float
    information_ratio: float
    tracking_error: float
    var_95: float
    cvar_95: float
    skewness: float
    kurtosis: float
    calmar_ratio: float
    omega_ratio: float

class PortfolioAnalytics:
    def __init__(self, config_path: str):
        """Initialize portfolio analytics with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Initialize Redis connection
        self.redis = redis.Redis(
            host='localhost',
            port=6379,
            db=0,
            decode_responses=True
        )
        
        # Storage keys
        self.metrics_key = 'portfolio:metrics'
        self.returns_key = 'portfolio:returns'
        self.positions_key = 'portfolio:positions'
        self.trades_key = 'portfolio:trades'
        
        # Load configuration
        self.risk_free_rate = self.config['trading']['risk_free_rate']
        self.benchmark_symbol = self.config['trading']['benchmark']
        
    def calculate_portfolio_metrics(self, positions: Dict[str, Dict], 
                                  trades: List[Dict],
                                  current_prices: Dict[str, float]) -> PortfolioMetrics:
        """Calculate comprehensive portfolio metrics"""
        # Calculate basic metrics
        positions_value = self._calculate_positions_value(positions, current_prices)
        cash = self._get_cash_balance()
        total_value = cash + positions_value
        
        # Calculate PnL metrics
        unrealized_pnl = self._calculate_unrealized_pnl(positions, current_prices)
        realized_pnl = self._calculate_realized_pnl(trades)
        total_pnl = unrealized_pnl + realized_pnl
        
        # Calculate returns
        returns = self._get_returns_history()
        daily_return = self._calculate_daily_return(returns)
        
        # Calculate risk metrics
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        max_drawdown = self._calculate_max_drawdown(returns)
        current_drawdown = self._calculate_current_drawdown(returns)
        
        # Calculate trade metrics
        win_rate = self._calculate_win_rate(trades)
        profit_factor = self._calculate_profit_factor(trades)
        average_win = self._calculate_average_win(trades)
        average_loss = self._calculate_average_loss(trades)
        largest_win = self._calculate_largest_win(trades)
        largest_loss = self._calculate_largest_loss(trades)
        average_trade_duration = self._calculate_average_trade_duration(trades)
        
        # Calculate position metrics
        position_count = len(positions)
        leverage = self._calculate_leverage(positions, current_prices)
        margin_level = self._calculate_margin_level(positions, current_prices)
        
        # Calculate advanced metrics
        beta = self._calculate_beta(returns)
        alpha = self._calculate_alpha(returns, beta)
        information_ratio = self._calculate_information_ratio(returns)
        tracking_error = self._calculate_tracking_error(returns)
        var_95 = self._calculate_var(returns, 0.95)
        cvar_95 = self._calculate_cvar(returns, 0.95)
        skewness = self._calculate_skewness(returns)
        kurtosis = self._calculate_kurtosis(returns)
        calmar_ratio = self._calculate_calmar_ratio(returns)
        omega_ratio = self._calculate_omega_ratio(returns)
        
        # Create metrics object
        metrics = PortfolioMetrics(
            total_value=total_value,
            cash=cash,
            positions_value=positions_value,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            total_pnl=total_pnl,
            daily_pnl=daily_return * total_value,
            daily_return=daily_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            average_win=average_win,
            average_loss=average_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            average_trade_duration=average_trade_duration,
            position_count=position_count,
            leverage=leverage,
            margin_level=margin_level,
            beta=beta,
            alpha=alpha,
            information_ratio=information_ratio,
            tracking_error=tracking_error,
            var_95=var_95,
            cvar_95=cvar_95,
            skewness=skewness,
            kurtosis=kurtosis,
            calmar_ratio=calmar_ratio,
            omega_ratio=omega_ratio
        )
        
        # Store metrics
        self._store_metrics(metrics)
        
        return metrics
        
    def generate_portfolio_report(self, metrics: PortfolioMetrics) -> Dict:
        """Generate comprehensive portfolio report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_value': metrics.total_value,
                'daily_pnl': metrics.daily_pnl,
                'daily_return': metrics.daily_return,
                'total_pnl': metrics.total_pnl,
                'position_count': metrics.position_count
            },
            'risk_metrics': {
                'sharpe_ratio': metrics.sharpe_ratio,
                'sortino_ratio': metrics.sortino_ratio,
                'max_drawdown': metrics.max_drawdown,
                'current_drawdown': metrics.current_drawdown,
                'var_95': metrics.var_95,
                'cvar_95': metrics.cvar_95,
                'beta': metrics.beta,
                'alpha': metrics.alpha
            },
            'trade_metrics': {
                'win_rate': metrics.win_rate,
                'profit_factor': metrics.profit_factor,
                'average_win': metrics.average_win,
                'average_loss': metrics.average_loss,
                'largest_win': metrics.largest_win,
                'largest_loss': metrics.largest_loss,
                'average_trade_duration': metrics.average_trade_duration
            },
            'position_metrics': {
                'leverage': metrics.leverage,
                'margin_level': metrics.margin_level,
                'unrealized_pnl': metrics.unrealized_pnl,
                'realized_pnl': metrics.realized_pnl
            },
            'advanced_metrics': {
                'information_ratio': metrics.information_ratio,
                'tracking_error': metrics.tracking_error,
                'skewness': metrics.skewness,
                'kurtosis': metrics.kurtosis,
                'calmar_ratio': metrics.calmar_ratio,
                'omega_ratio': metrics.omega_ratio
            }
        }
        
        # Store report
        self._store_report(report)
        
        return report
        
    def _calculate_positions_value(self, positions: Dict[str, Dict], 
                                 current_prices: Dict[str, float]) -> float:
        """Calculate total value of all positions"""
        return sum(
            position['quantity'] * current_prices[symbol]
            for symbol, position in positions.items()
        )
        
    def _get_cash_balance(self) -> float:
        """Get current cash balance from Redis"""
        cash_key = 'trading:cash'
        cash = self.redis.get(cash_key)
        return float(cash) if cash else 0.0
        
    def _calculate_unrealized_pnl(self, positions: Dict[str, Dict],
                                current_prices: Dict[str, float]) -> float:
        """Calculate unrealized PnL for all positions"""
        return sum(
            (current_prices[symbol] - position['average_price']) * position['quantity']
            for symbol, position in positions.items()
        )
        
    def _calculate_realized_pnl(self, trades: List[Dict]) -> float:
        """Calculate realized PnL from trades"""
        return sum(trade.get('pnl', 0) for trade in trades)
        
    def _get_returns_history(self) -> np.ndarray:
        """Get historical returns from Redis"""
        returns_key = self.returns_key
        returns = self.redis.lrange(returns_key, 0, -1)
        return np.array([float(r) for r in returns])
        
    def _calculate_daily_return(self, returns: np.ndarray) -> float:
        """Calculate daily return"""
        return returns[-1] if len(returns) > 0 else 0.0
        
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0.0
        excess_returns = returns - self.risk_free_rate / 252
        return np.mean(excess_returns) / (np.std(excess_returns) + 1e-6) * np.sqrt(252)
        
    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio"""
        if len(returns) < 2:
            return 0.0
        excess_returns = returns - self.risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return float('inf')
        return np.mean(excess_returns) / (np.std(downside_returns) + 1e-6) * np.sqrt(252)
        
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        if len(returns) < 2:
            return 0.0
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (running_max - cumulative_returns) / running_max
        return np.max(drawdowns)
        
    def _calculate_current_drawdown(self, returns: np.ndarray) -> float:
        """Calculate current drawdown"""
        if len(returns) < 2:
            return 0.0
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (running_max - cumulative_returns) / running_max
        return drawdowns[-1]
        
    def _calculate_win_rate(self, trades: List[Dict]) -> float:
        """Calculate win rate"""
        if not trades:
            return 0.0
        winning_trades = sum(1 for t in trades if t.get('pnl', 0) > 0)
        return winning_trades / len(trades)
        
    def _calculate_profit_factor(self, trades: List[Dict]) -> float:
        """Calculate profit factor"""
        if not trades:
            return 0.0
        gross_profit = sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0)
        gross_loss = abs(sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0))
        return gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
    def _calculate_average_win(self, trades: List[Dict]) -> float:
        """Calculate average winning trade"""
        winning_trades = [t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0]
        return np.mean(winning_trades) if winning_trades else 0.0
        
    def _calculate_average_loss(self, trades: List[Dict]) -> float:
        """Calculate average losing trade"""
        losing_trades = [t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0]
        return np.mean(losing_trades) if losing_trades else 0.0
        
    def _calculate_largest_win(self, trades: List[Dict]) -> float:
        """Calculate largest winning trade"""
        winning_trades = [t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0]
        return max(winning_trades) if winning_trades else 0.0
        
    def _calculate_largest_loss(self, trades: List[Dict]) -> float:
        """Calculate largest losing trade"""
        losing_trades = [t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0]
        return min(losing_trades) if losing_trades else 0.0
        
    def _calculate_average_trade_duration(self, trades: List[Dict]) -> float:
        """Calculate average trade duration in hours"""
        if not trades:
            return 0.0
        durations = []
        for trade in trades:
            if 'entry_time' in trade and 'exit_time' in trade:
                entry = datetime.fromisoformat(trade['entry_time'])
                exit = datetime.fromisoformat(trade['exit_time'])
                duration = (exit - entry).total_seconds() / 3600
                durations.append(duration)
        return np.mean(durations) if durations else 0.0
        
    def _calculate_leverage(self, positions: Dict[str, Dict],
                          current_prices: Dict[str, float]) -> float:
        """Calculate current leverage"""
        total_value = self._calculate_positions_value(positions, current_prices)
        cash = self._get_cash_balance()
        return total_value / cash if cash > 0 else 0.0
        
    def _calculate_margin_level(self, positions: Dict[str, Dict],
                              current_prices: Dict[str, float]) -> float:
        """Calculate current margin level"""
        total_value = self._calculate_positions_value(positions, current_prices)
        cash = self._get_cash_balance()
        used_margin = sum(p.get('margin_used', 0) for p in positions.values())
        return (total_value + cash) / used_margin * 100 if used_margin > 0 else 100
        
    def _calculate_beta(self, returns: np.ndarray) -> float:
        """Calculate portfolio beta"""
        if len(returns) < 2:
            return 0.0
        benchmark_returns = self._get_benchmark_returns()
        if len(benchmark_returns) != len(returns):
            return 0.0
        covariance = np.cov(returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        return covariance / benchmark_variance if benchmark_variance != 0 else 0.0
        
    def _calculate_alpha(self, returns: np.ndarray, beta: float) -> float:
        """Calculate portfolio alpha"""
        if len(returns) < 2:
            return 0.0
        benchmark_returns = self._get_benchmark_returns()
        if len(benchmark_returns) != len(returns):
            return 0.0
        excess_returns = returns - self.risk_free_rate / 252
        excess_benchmark = benchmark_returns - self.risk_free_rate / 252
        return np.mean(excess_returns) - beta * np.mean(excess_benchmark)
        
    def _calculate_information_ratio(self, returns: np.ndarray) -> float:
        """Calculate information ratio"""
        if len(returns) < 2:
            return 0.0
        benchmark_returns = self._get_benchmark_returns()
        if len(benchmark_returns) != len(returns):
            return 0.0
        active_returns = returns - benchmark_returns
        return np.mean(active_returns) / (np.std(active_returns) + 1e-6) * np.sqrt(252)
        
    def _calculate_tracking_error(self, returns: np.ndarray) -> float:
        """Calculate tracking error"""
        if len(returns) < 2:
            return 0.0
        benchmark_returns = self._get_benchmark_returns()
        if len(benchmark_returns) != len(returns):
            return 0.0
        active_returns = returns - benchmark_returns
        return np.std(active_returns) * np.sqrt(252)
        
    def _calculate_var(self, returns: np.ndarray, confidence: float) -> float:
        """Calculate Value at Risk"""
        if len(returns) < 2:
            return 0.0
        return np.percentile(returns, (1 - confidence) * 100)
        
    def _calculate_cvar(self, returns: np.ndarray, confidence: float) -> float:
        """Calculate Conditional Value at Risk"""
        if len(returns) < 2:
            return 0.0
        var = self._calculate_var(returns, confidence)
        return np.mean(returns[returns <= var])
        
    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Calculate returns skewness"""
        if len(returns) < 2:
            return 0.0
        return stats.skew(returns)
        
    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """Calculate returns kurtosis"""
        if len(returns) < 2:
            return 0.0
        return stats.kurtosis(returns)
        
    def _calculate_calmar_ratio(self, returns: np.ndarray) -> float:
        """Calculate Calmar ratio"""
        if len(returns) < 2:
            return 0.0
        annual_return = np.mean(returns) * 252
        max_drawdown = self._calculate_max_drawdown(returns)
        return annual_return / max_drawdown if max_drawdown != 0 else float('inf')
        
    def _calculate_omega_ratio(self, returns: np.ndarray) -> float:
        """Calculate Omega ratio"""
        if len(returns) < 2:
            return 0.0
        threshold = self.risk_free_rate / 252
        gains = returns[returns > threshold]
        losses = returns[returns <= threshold]
        if len(losses) == 0:
            return float('inf')
        return np.sum(gains - threshold) / abs(np.sum(losses - threshold))
        
    def _get_benchmark_returns(self) -> np.ndarray:
        """Get benchmark returns from Redis"""
        benchmark_key = f"returns:{self.benchmark_symbol}"
        returns = self.redis.lrange(benchmark_key, 0, -1)
        return np.array([float(r) for r in returns])
        
    def _store_metrics(self, metrics: PortfolioMetrics) -> None:
        """Store metrics in Redis"""
        metrics_key = f"{self.metrics_key}:{datetime.now().isoformat()}"
        self.redis.setex(
            metrics_key,
            timedelta(days=1),
            json.dumps(metrics.__dict__)
        )
        self.redis.lpush(self.metrics_key, metrics_key)
        self.redis.ltrim(self.metrics_key, 0, 1439)  # Keep last 24 hours
        
    def _store_report(self, report: Dict) -> None:
        """Store report in Redis"""
        report_key = f"report:{datetime.now().isoformat()}"
        self.redis.setex(
            report_key,
            timedelta(days=30),
            json.dumps(report)
        ) 