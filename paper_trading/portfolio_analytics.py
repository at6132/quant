import logging
from datetime import datetime
from typing import Dict, List, Optional, Union
import yaml
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PortfolioAnalytics:
    def __init__(self, config: Union[str, Dict] = "paper_trading_config.yaml"):
        """Initialize portfolio analytics with configuration
        
        Args:
            config: Either a path to the config file or a config dictionary
        """
        if isinstance(config, str):
            with open(config, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config
            
        # Load settings
        self.account_settings = self.config.get('account', {})
        self.risk_settings = self.config.get('risk_management', {})
        
        # Initialize state
        self.metrics_history = []
        self.reports = []
        
        logger.info("Portfolio analytics initialized")
        
    def calculate_portfolio_metrics(self, positions: Dict, trades: List[Dict], prices: Dict) -> Dict:
        """Calculate portfolio metrics"""
        try:
            # Calculate total value
            total_value = self._calculate_total_value(positions, prices)
            
            # Calculate cash
            cash = self._calculate_cash(positions, trades)
            
            # Calculate margin used
            margin_used = self._calculate_margin_used(positions, prices)
            
            # Calculate leverage
            leverage = margin_used / total_value if total_value > 0 else 0
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(positions, trades, prices)
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(trades)
            
            # Calculate trade statistics
            trade_stats = self._calculate_trade_statistics(trades)
            
            # Calculate PnL metrics
            pnl_metrics = self._calculate_pnl_metrics(trades)
            
            # Combine all metrics
            metrics = {
                'total_value': total_value,
                'cash': cash,
                'margin_used': margin_used,
                'leverage': leverage,
                **risk_metrics,
                **performance_metrics,
                **trade_stats,
                **pnl_metrics
            }
            
            return metrics
            
        except Exception as e:
            logger.error("Error calculating portfolio metrics: %s", str(e))
            return {}
            
    def _calculate_total_value(self, positions: Dict, prices: Dict) -> float:
        """Calculate total portfolio value"""
        try:
            position_value = sum(
                position['size'] * prices.get(symbol, 0)
                for symbol, position in positions.items()
            )
            return position_value
        except Exception as e:
            logger.error("Error calculating total value: %s", str(e))
            return 0.0
            
    def _calculate_cash(self, positions: Dict, trades: List[Dict]) -> float:
        """Calculate available cash"""
        try:
            # Start with initial capital
            cash = self.account_settings.get('initial_balance', 100000.0)
            
            # Subtract margin used
            margin_used = self._calculate_margin_used(positions, {})
            cash -= margin_used
            
            return cash
        except Exception as e:
            logger.error("Error calculating cash: %s", str(e))
            return 0.0
            
    def _calculate_margin_used(self, positions: Dict, prices: Dict) -> float:
        """Calculate margin used by positions"""
        try:
            margin_used = sum(
                position['size'] * prices.get(symbol, 0) / self.account_settings.get('max_leverage', 3.0)
                for symbol, position in positions.items()
            )
            return margin_used
        except Exception as e:
            logger.error("Error calculating margin used: %s", str(e))
            return 0.0
            
    def _calculate_risk_metrics(self, positions: Dict, trades: List[Dict], prices: Dict) -> Dict:
        """Calculate risk metrics"""
        try:
            # Calculate drawdown
            drawdown = self._calculate_drawdown(trades)
            
            # Calculate VaR
            var_95 = self._calculate_var(trades)
            
            # Calculate volatility
            volatility = self._calculate_volatility(trades)
            
            return {
                'max_drawdown': drawdown,
                'var_95': var_95,
                'volatility': volatility
            }
        except Exception as e:
            logger.error("Error calculating risk metrics: %s", str(e))
            return {}
            
    def _calculate_performance_metrics(self, trades: List[Dict]) -> Dict:
        """Calculate performance metrics"""
        try:
            if not trades:
                return {
                    'sharpe_ratio': 0.0,
                    'sortino_ratio': 0.0,
                    'win_rate': 0.0
                }
                
            # Calculate returns
            returns = [trade.get('pnl', 0) for trade in trades]
            
            # Calculate Sharpe ratio
            sharpe = self._calculate_sharpe_ratio(returns)
            
            # Calculate Sortino ratio
            sortino = self._calculate_sortino_ratio(returns)
            
            # Calculate win rate
            win_rate = len([r for r in returns if r > 0]) / len(returns) if returns else 0
            
            return {
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino,
                'win_rate': win_rate
            }
        except Exception as e:
            logger.error("Error calculating performance metrics: %s", str(e))
            return {}
            
    def _calculate_trade_statistics(self, trades: List[Dict]) -> Dict:
        """Calculate trade statistics"""
        try:
            if not trades:
                return {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'average_trade': 0.0
                }
                
            # Calculate basic statistics
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t.get('pnl', 0) > 0])
            losing_trades = total_trades - winning_trades
            average_trade = sum(t.get('pnl', 0) for t in trades) / total_trades if total_trades > 0 else 0
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'average_trade': average_trade
            }
        except Exception as e:
            logger.error("Error calculating trade statistics: %s", str(e))
            return {}
            
    def _calculate_pnl_metrics(self, trades: List[Dict]) -> Dict:
        """Calculate PnL metrics"""
        try:
            if not trades:
                return {
                    'total_pnl': 0.0,
                    'realized_pnl': 0.0,
                    'unrealized_pnl': 0.0
                }
                
            # Calculate PnL metrics
            total_pnl = sum(t.get('pnl', 0) for t in trades)
            realized_pnl = sum(t.get('pnl', 0) for t in trades if t.get('status') == 'closed')
            unrealized_pnl = total_pnl - realized_pnl
            
            return {
                'total_pnl': total_pnl,
                'realized_pnl': realized_pnl,
                'unrealized_pnl': unrealized_pnl
            }
        except Exception as e:
            logger.error("Error calculating PnL metrics: %s", str(e))
            return {}
            
    def _calculate_drawdown(self, trades: List[Dict]) -> float:
        """Calculate maximum drawdown"""
        try:
            if not trades:
                return 0.0
                
            # Calculate cumulative returns
            returns = [t.get('pnl', 0) for t in trades]
            cumulative = np.cumsum(returns)
            
            # Calculate drawdown
            max_dd = 0
            peak = cumulative[0]
            
            for value in cumulative:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)
                
            return max_dd
        except Exception as e:
            logger.error("Error calculating drawdown: %s", str(e))
            return 0.0
            
    def _calculate_var(self, trades: List[Dict]) -> float:
        """Calculate Value at Risk at 95% confidence"""
        try:
            if not trades:
                return 0.0
                
            # Calculate returns
            returns = [t.get('pnl', 0) for t in trades]
            
            # Calculate VaR
            var_95 = np.percentile(returns, 5)
            return abs(var_95)
        except Exception as e:
            logger.error("Error calculating VaR: %s", str(e))
            return 0.0
            
    def _calculate_volatility(self, trades: List[Dict]) -> float:
        """Calculate portfolio volatility"""
        try:
            if not trades:
                return 0.0
                
            # Calculate returns
            returns = [t.get('pnl', 0) for t in trades]
            
            # Calculate volatility
            volatility = np.std(returns)
            return volatility
        except Exception as e:
            logger.error("Error calculating volatility: %s", str(e))
            return 0.0
            
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio"""
        try:
            if not returns:
                return 0.0
                
            # Calculate excess returns
            excess_returns = [r - self.risk_settings.get('risk_free_rate', 0.02)/252 for r in returns]  # Daily risk-free rate
            
            # Calculate Sharpe ratio
            if len(excess_returns) > 1:
                sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
                return sharpe
            return 0.0
        except Exception as e:
            logger.error("Error calculating Sharpe ratio: %s", str(e))
            return 0.0
            
    def _calculate_sortino_ratio(self, returns: List[float]) -> float:
        """Calculate Sortino ratio"""
        try:
            if not returns:
                return 0.0
                
            # Calculate excess returns
            excess_returns = [r - self.risk_settings.get('risk_free_rate', 0.02)/252 for r in returns]  # Daily risk-free rate
            
            # Calculate downside returns
            downside_returns = [r for r in excess_returns if r < 0]
            
            # Calculate Sortino ratio
            if len(downside_returns) > 1:
                sortino = np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)
                return sortino
            return 0.0
        except Exception as e:
            logger.error("Error calculating Sortino ratio: %s", str(e))
            return 0.0
            
    def generate_portfolio_report(self, metrics: Dict) -> Dict:
        """Generate portfolio report"""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics,
                'thresholds': self.risk_settings.get('performance_thresholds', {
                    'min_sharpe': 1.0,
                    'min_sortino': 1.5,
                    'max_drawdown': 0.15,
                    'min_win_rate': 0.5
                }),
                'summary': self._generate_summary(metrics)
            }
            return report
        except Exception as e:
            logger.error("Error generating portfolio report: %s", str(e))
            return {}
            
    def _generate_summary(self, metrics: Dict) -> str:
        """Generate summary of portfolio performance"""
        try:
            summary = []
            
            # Add key metrics
            summary.append(f"Total Value: ${metrics.get('total_value', 0):,.2f}")
            summary.append(f"Cash: ${metrics.get('cash', 0):,.2f}")
            summary.append(f"Margin Used: ${metrics.get('margin_used', 0):,.2f}")
            summary.append(f"Leverage: {metrics.get('leverage', 0):.2f}x")
            
            # Add performance metrics
            summary.append(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            summary.append(f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}")
            summary.append(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
            
            # Add risk metrics
            summary.append(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
            summary.append(f"VaR(95%): {metrics.get('var_95', 0):.2%}")
            summary.append(f"Volatility: {metrics.get('volatility', 0):.2%}")
            
            # Add PnL metrics
            summary.append(f"Total PnL: ${metrics.get('total_pnl', 0):,.2f}")
            summary.append(f"Realized PnL: ${metrics.get('realized_pnl', 0):,.2f}")
            summary.append(f"Unrealized PnL: ${metrics.get('unrealized_pnl', 0):,.2f}")
            
            return "\n".join(summary)
        except Exception as e:
            logger.error("Error generating summary: %s", str(e))
            return "Error generating summary" 