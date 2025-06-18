"""
Performance Monitor - Real-time Performance Tracking and Alerting
Implements real-time metrics tracking, alert system, and performance reporting
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
from pathlib import Path

from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    timestamp: datetime
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    current_drawdown: float
    total_exposure: float
    risk_score: float
    model_accuracy: float
    confidence_score: float


@dataclass
class Alert:
    """Alert data structure"""
    timestamp: datetime
    level: str  # 'info', 'warning', 'error', 'critical'
    category: str
    message: str
    data: Dict[str, Any]


class PerformanceMonitor:
    """
    Real-time Performance Monitoring System
    """
    
    def __init__(self, config_path: str = "config/intelligent_config.yaml"):
        """Initialize performance monitor"""
        self.logger = logger
        
        # Configuration
        self.monitor_config = {
            'update_frequency': 60,  # seconds
            'metrics_window': 100,   # number of trades
            'alert_thresholds': {
                'consecutive_losses': 5,
                'drawdown_threshold': 0.15,
                'win_rate_threshold': 0.4,
                'profit_factor_threshold': 1.0,
                'exposure_threshold': 0.8
            }
        }
        
        # Performance tracking
        self.trade_history = []
        self.metrics_history = []
        self.alerts_history = []
        
        # Current metrics
        self.current_metrics = None
        
        # Alert callbacks
        self.alert_callbacks = []
        
        # Performance tracking
        self.start_time = datetime.now()
        self.peak_balance = 0.0
        self.current_balance = 0.0
        
        # Training timing
        self.training_start_time = None
        self.training_end_time = None
        
        self.logger.info("Performance Monitor initialized successfully")
    
    def start(self):
        """Start timing for training or other operations"""
        self.training_start_time = time.time()
        self.logger.info("Performance monitor timing started")
    
    def stop(self):
        """Stop timing for training or other operations"""
        self.training_end_time = time.time()
        self.logger.info("Performance monitor timing stopped")
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time since start() was called"""
        if self.training_start_time is None:
            return 0.0
        if self.training_end_time is not None:
            return self.training_end_time - self.training_start_time
        return time.time() - self.training_start_time
    
    async def update_metrics(
        self,
        trade_data: Dict[str, Any] = None,
        model_predictions: Dict[str, float] = None,
        risk_metrics: Dict[str, Any] = None,
        market_context: Dict[str, Any] = None
    ) -> PerformanceMetrics:
        """
        Update performance metrics with new data
        
        Args:
            trade_data: New trade data
            model_predictions: Model predictions
            risk_metrics: Risk metrics
            market_context: Market context
            
        Returns:
            Updated PerformanceMetrics object
        """
        try:
            # Update trade history
            if trade_data:
                self.trade_history.append(trade_data)
                
                # Update balance tracking
                pnl = trade_data.get('pnl', 0)
                self.current_balance += pnl
                self.peak_balance = max(self.peak_balance, self.current_balance)
            
            # Calculate metrics
            metrics = await self._calculate_metrics(model_predictions, risk_metrics, market_context)
            
            # Store metrics
            self.metrics_history.append(metrics)
            self.current_metrics = metrics
            
            # Check for alerts
            await self._check_alerts(metrics)
            
            # Limit history size
            max_history = self.monitor_config['metrics_window']
            if len(self.metrics_history) > max_history:
                self.metrics_history = self.metrics_history[-max_history:]
            
            if len(self.trade_history) > max_history:
                self.trade_history = self.trade_history[-max_history:]
            
            self.logger.info(f"Performance metrics updated - Win rate: {metrics.win_rate:.2%}, P&L: {metrics.total_pnl:.2f}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
            return await self._create_default_metrics()
    
    async def _calculate_metrics(
        self,
        model_predictions: Dict[str, float] = None,
        risk_metrics: Dict[str, Any] = None,
        market_context: Dict[str, Any] = None
    ) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        try:
            if not self.trade_history:
                return await self._create_default_metrics()
            
            # Basic trade metrics
            total_trades = len(self.trade_history)
            winning_trades = sum(1 for trade in self.trade_history if trade.get('pnl', 0) > 0)
            losing_trades = total_trades - winning_trades
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            
            # P&L metrics
            total_pnl = sum(trade.get('pnl', 0) for trade in self.trade_history)
            
            winning_pnls = [trade['pnl'] for trade in self.trade_history if trade.get('pnl', 0) > 0]
            losing_pnls = [trade['pnl'] for trade in self.trade_history if trade.get('pnl', 0) < 0]
            
            avg_win = np.mean(winning_pnls) if winning_pnls else 0.0
            avg_loss = abs(np.mean(losing_pnls)) if losing_pnls else 0.0
            
            # Profit factor
            gross_profit = sum(winning_pnls) if winning_pnls else 0.0
            gross_loss = abs(sum(losing_pnls)) if losing_pnls else 0.0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Sharpe ratio (simplified)
            returns = [trade.get('pnl', 0) for trade in self.trade_history]
            if len(returns) > 1:
                sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0
            else:
                sharpe_ratio = 0.0
            
            # Drawdown calculation
            cumulative_pnl = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative_pnl)
            drawdowns = (running_max - cumulative_pnl) / running_max if running_max[-1] > 0 else 0
            max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
            current_drawdown = drawdowns[-1] if len(drawdowns) > 0 else 0.0
            
            # Risk metrics
            total_exposure = risk_metrics.get('total_exposure', 0.0) if risk_metrics else 0.0
            risk_score = risk_metrics.get('risk_score', 0.5) if risk_metrics else 0.5
            
            # Model accuracy
            model_accuracy = 0.0
            if model_predictions:
                # Calculate accuracy based on predictions vs actual outcomes
                correct_predictions = 0
                total_predictions = 0
                
                for trade in self.trade_history[-20:]:  # Last 20 trades
                    if 'predicted_direction' in trade and 'actual_direction' in trade:
                        if trade['predicted_direction'] == trade['actual_direction']:
                            correct_predictions += 1
                        total_predictions += 1
                
                if total_predictions > 0:
                    model_accuracy = correct_predictions / total_predictions
            
            # Confidence score
            confidence_score = 0.0
            if model_predictions:
                confidence_score = model_predictions.get('entry_confidence', 0.5)
            
            # Create metrics object
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                total_pnl=total_pnl,
                avg_win=avg_win,
                avg_loss=avg_loss,
                profit_factor=profit_factor,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                current_drawdown=current_drawdown,
                total_exposure=total_exposure,
                risk_score=risk_score,
                model_accuracy=model_accuracy,
                confidence_score=confidence_score
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            return await self._create_default_metrics()
    
    async def _check_alerts(self, metrics: PerformanceMetrics):
        """Check for performance alerts"""
        try:
            thresholds = self.monitor_config['alert_thresholds']
            
            # Consecutive losses alert
            consecutive_losses = await self._get_consecutive_losses()
            if consecutive_losses >= thresholds['consecutive_losses']:
                await self._create_alert(
                    level='warning',
                    category='performance',
                    message=f"Consecutive losses: {consecutive_losses}",
                    data={'consecutive_losses': consecutive_losses}
                )
            
            # Drawdown alert
            if metrics.current_drawdown >= thresholds['drawdown_threshold']:
                await self._create_alert(
                    level='warning',
                    category='risk',
                    message=f"High drawdown: {metrics.current_drawdown:.2%}",
                    data={'current_drawdown': metrics.current_drawdown}
                )
            
            # Win rate alert
            if metrics.win_rate <= thresholds['win_rate_threshold']:
                await self._create_alert(
                    level='warning',
                    category='performance',
                    message=f"Low win rate: {metrics.win_rate:.2%}",
                    data={'win_rate': metrics.win_rate}
                )
            
            # Profit factor alert
            if metrics.profit_factor <= thresholds['profit_factor_threshold']:
                await self._create_alert(
                    level='warning',
                    category='performance',
                    message=f"Low profit factor: {metrics.profit_factor:.2f}",
                    data={'profit_factor': metrics.profit_factor}
                )
            
            # Exposure alert
            if metrics.total_exposure >= thresholds['exposure_threshold']:
                await self._create_alert(
                    level='warning',
                    category='risk',
                    message=f"High exposure: {metrics.total_exposure:.2%}",
                    data={'total_exposure': metrics.total_exposure}
                )
            
            # Critical alerts
            if metrics.current_drawdown >= 0.25:  # 25% drawdown
                await self._create_alert(
                    level='critical',
                    category='risk',
                    message=f"Critical drawdown: {metrics.current_drawdown:.2%}",
                    data={'current_drawdown': metrics.current_drawdown}
                )
            
            if consecutive_losses >= 10:  # 10 consecutive losses
                await self._create_alert(
                    level='critical',
                    category='performance',
                    message=f"Critical consecutive losses: {consecutive_losses}",
                    data={'consecutive_losses': consecutive_losses}
                )
            
        except Exception as e:
            self.logger.error(f"Error checking alerts: {e}")
    
    async def _get_consecutive_losses(self) -> int:
        """Get number of consecutive losses"""
        try:
            consecutive_losses = 0
            for trade in reversed(self.trade_history):
                if trade.get('pnl', 0) < 0:
                    consecutive_losses += 1
                else:
                    break
            return consecutive_losses
        except Exception as e:
            self.logger.error(f"Error calculating consecutive losses: {e}")
            return 0
    
    async def _create_alert(
        self,
        level: str,
        category: str,
        message: str,
        data: Dict[str, Any]
    ):
        """Create and store an alert"""
        try:
            alert = Alert(
                timestamp=datetime.now(),
                level=level,
                category=category,
                message=message,
                data=data
            )
            
            self.alerts_history.append(alert)
            
            # Log alert
            if level == 'critical':
                self.logger.critical(f"CRITICAL ALERT: {message}")
            elif level == 'error':
                self.logger.error(f"ERROR ALERT: {message}")
            elif level == 'warning':
                self.logger.warning(f"WARNING ALERT: {message}")
            else:
                self.logger.info(f"INFO ALERT: {message}")
            
            # Call alert callbacks
            for callback in self.alert_callbacks:
                try:
                    await callback(alert)
                except Exception as e:
                    self.logger.error(f"Error in alert callback: {e}")
            
        except Exception as e:
            self.logger.error(f"Error creating alert: {e}")
    
    def add_alert_callback(self, callback):
        """Add alert callback function"""
        self.alert_callbacks.append(callback)
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        try:
            if not self.current_metrics:
                return await self._create_default_summary()
            
            # Calculate additional metrics
            uptime = datetime.now() - self.start_time
            trades_per_hour = self.current_metrics.total_trades / (uptime.total_seconds() / 3600) if uptime.total_seconds() > 0 else 0
            
            # Recent performance (last 20 trades)
            recent_trades = self.trade_history[-20:] if len(self.trade_history) >= 20 else self.trade_history
            recent_win_rate = sum(1 for trade in recent_trades if trade.get('pnl', 0) > 0) / len(recent_trades) if recent_trades else 0
            
            summary = {
                'current_metrics': {
                    'total_trades': self.current_metrics.total_trades,
                    'win_rate': self.current_metrics.win_rate,
                    'total_pnl': self.current_metrics.total_pnl,
                    'profit_factor': self.current_metrics.profit_factor,
                    'sharpe_ratio': self.current_metrics.sharpe_ratio,
                    'max_drawdown': self.current_metrics.max_drawdown,
                    'current_drawdown': self.current_metrics.current_drawdown,
                    'model_accuracy': self.current_metrics.model_accuracy,
                    'confidence_score': self.current_metrics.confidence_score
                },
                'system_info': {
                    'uptime': str(uptime),
                    'start_time': self.start_time.isoformat(),
                    'trades_per_hour': trades_per_hour,
                    'recent_win_rate': recent_win_rate
                },
                'risk_metrics': {
                    'total_exposure': self.current_metrics.total_exposure,
                    'risk_score': self.current_metrics.risk_score
                },
                'alerts': {
                    'total_alerts': len(self.alerts_history),
                    'critical_alerts': len([a for a in self.alerts_history if a.level == 'critical']),
                    'warning_alerts': len([a for a in self.alerts_history if a.level == 'warning']),
                    'recent_alerts': [a.__dict__ for a in self.alerts_history[-5:]]
                }
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting performance summary: {e}")
            return await self._create_default_summary()
    
    async def generate_performance_report(self, report_type: str = 'daily') -> str:
        """Generate performance report"""
        try:
            if not self.current_metrics:
                return "No performance data available"
            
            report = f"Performance Report - {report_type.title()}\n"
            report += "=" * 50 + "\n\n"
            
            # Basic metrics
            report += f"Total Trades: {self.current_metrics.total_trades}\n"
            report += f"Win Rate: {self.current_metrics.win_rate:.2%}\n"
            report += f"Total P&L: ${self.current_metrics.total_pnl:.2f}\n"
            report += f"Profit Factor: {self.current_metrics.profit_factor:.2f}\n"
            report += f"Sharpe Ratio: {self.current_metrics.sharpe_ratio:.2f}\n"
            report += f"Max Drawdown: {self.current_metrics.max_drawdown:.2%}\n"
            report += f"Current Drawdown: {self.current_metrics.current_drawdown:.2%}\n"
            report += f"Model Accuracy: {self.current_metrics.model_accuracy:.2%}\n"
            report += f"Confidence Score: {self.current_metrics.confidence_score:.2%}\n\n"
            
            # Risk metrics
            report += f"Total Exposure: {self.current_metrics.total_exposure:.2%}\n"
            report += f"Risk Score: {self.current_metrics.risk_score:.3f}\n\n"
            
            # Alerts summary
            critical_alerts = len([a for a in self.alerts_history if a.level == 'critical'])
            warning_alerts = len([a for a in self.alerts_history if a.level == 'warning'])
            report += f"Critical Alerts: {critical_alerts}\n"
            report += f"Warning Alerts: {warning_alerts}\n\n"
            
            # Recent alerts
            if self.alerts_history:
                report += "Recent Alerts:\n"
                for alert in self.alerts_history[-5:]:
                    report += f"  [{alert.level.upper()}] {alert.message}\n"
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating performance report: {e}")
            return f"Error generating report: {str(e)}"
    
    async def save_metrics_to_file(self, filepath: str = "logs/performance_metrics.json"):
        """Save metrics to JSON file"""
        try:
            if not self.current_metrics:
                return
            
            # Create directory if it doesn't exist
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare data for saving
            data = {
                'timestamp': self.current_metrics.timestamp.isoformat(),
                'metrics': self.current_metrics.__dict__,
                'trade_history': self.trade_history[-50:],  # Last 50 trades
                'alerts': [alert.__dict__ for alert in self.alerts_history[-20:]]  # Last 20 alerts
            }
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            self.logger.info(f"Performance metrics saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving metrics to file: {e}")
    
    async def _create_default_metrics(self) -> PerformanceMetrics:
        """Create default performance metrics"""
        return PerformanceMetrics(
            timestamp=datetime.now(),
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_pnl=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            profit_factor=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            current_drawdown=0.0,
            total_exposure=0.0,
            risk_score=0.5,
            model_accuracy=0.0,
            confidence_score=0.5
        )
    
    async def _create_default_summary(self) -> Dict[str, Any]:
        """Create default performance summary"""
        return {
            'current_metrics': {},
            'system_info': {
                'uptime': '0:00:00',
                'start_time': self.start_time.isoformat(),
                'trades_per_hour': 0.0,
                'recent_win_rate': 0.0
            },
            'risk_metrics': {
                'total_exposure': 0.0,
                'risk_score': 0.5
            },
            'alerts': {
                'total_alerts': 0,
                'critical_alerts': 0,
                'warning_alerts': 0,
                'recent_alerts': []
            }
        }
    
    def get_metrics_history(self) -> List[PerformanceMetrics]:
        """Get metrics history"""
        return self.metrics_history.copy()
    
    def get_alerts_history(self) -> List[Alert]:
        """Get alerts history"""
        return self.alerts_history.copy()
    
    def reset(self):
        """Reset performance monitor (for testing)"""
        self.trade_history.clear()
        self.metrics_history.clear()
        self.alerts_history.clear()
        self.current_metrics = None
        self.start_time = datetime.now()
        self.peak_balance = 0.0
        self.current_balance = 0.0 