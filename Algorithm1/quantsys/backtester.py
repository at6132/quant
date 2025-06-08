import pandas as pd
import numpy as np
from typing import Dict, Tuple
import matplotlib.pyplot as plt
from datetime import datetime

class Backtester:
    def __init__(self, 
                 initial_capital: float = 100000,
                 position_size: float = 0.1,  # 10% of capital per trade
                 transaction_cost: float = 0.001,  # 0.1% per trade
                 slippage: float = 0.0005):  # 0.05% slippage
        
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        
    def calculate_returns(self, signals: pd.DataFrame, price_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate returns based on signals."""
        results = pd.DataFrame(index=signals.index)
        
        # Get prices
        results['close'] = price_data['close']
        results['signal'] = signals['signal']
        
        # Calculate position changes
        results['position'] = results['signal'].fillna(0)
        results['position_change'] = results['position'].diff().fillna(0)
        
        # Calculate returns
        results['returns'] = results['close'].pct_change().fillna(0)
        results['strategy_returns'] = results['position'].shift(1) * results['returns']
        
        # Apply transaction costs and slippage
        # Cost when entering or exiting position
        trade_mask = results['position_change'] != 0
        cost_per_trade = self.transaction_cost + self.slippage
        
        results.loc[trade_mask, 'strategy_returns'] -= cost_per_trade
        
        # Calculate cumulative returns
        results['cumulative_returns'] = (1 + results['returns']).cumprod()
        results['cumulative_strategy_returns'] = (1 + results['strategy_returns']).cumprod()
        
        # Calculate equity curve
        results['equity'] = self.initial_capital * results['cumulative_strategy_returns']
        
        return results
    
    def calculate_metrics(self, results: pd.DataFrame) -> Dict:
        """Calculate performance metrics."""
        # Basic returns
        total_return = results['cumulative_strategy_returns'].iloc[-1] - 1
        buy_hold_return = results['cumulative_returns'].iloc[-1] - 1
        
        # Annualized returns (assuming 15-second bars)
        periods_per_year = 252 * 24 * 60 * 4  # Trading days * hours * minutes * 4
        n_periods = len(results)
        years = n_periods / periods_per_year
        
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        annualized_buy_hold = (1 + buy_hold_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Volatility
        strategy_vol = results['strategy_returns'].std() * np.sqrt(periods_per_year)
        buy_hold_vol = results['returns'].std() * np.sqrt(periods_per_year)
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = annualized_return / strategy_vol if strategy_vol > 0 else 0
        
        # Maximum drawdown
        equity = results['equity']
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_trades = results.loc[results['strategy_returns'] > 0, 'strategy_returns']
        losing_trades = results.loc[results['strategy_returns'] < 0, 'strategy_returns']
        
        n_trades = (results['position_change'] != 0).sum()
        win_rate = len(winning_trades) / n_trades if n_trades > 0 else 0
        
        # Average win/loss
        avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0
        
        # Profit factor
        total_wins = winning_trades.sum() if len(winning_trades) > 0 else 0
        total_losses = abs(losing_trades.sum()) if len(losing_trades) > 0 else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        metrics = {
            'Total Return': f"{total_return:.2%}",
            'Buy & Hold Return': f"{buy_hold_return:.2%}",
            'Annualized Return': f"{annualized_return:.2%}",
            'Annualized Vol': f"{strategy_vol:.2%}",
            'Sharpe Ratio': f"{sharpe_ratio:.2f}",
            'Max Drawdown': f"{max_drawdown:.2%}",
            'Calmar Ratio': f"{calmar_ratio:.2f}",
            'Win Rate': f"{win_rate:.2%}",
            'Avg Win': f"{avg_win:.2%}",
            'Avg Loss': f"{avg_loss:.2%}",
            'Profit Factor': f"{profit_factor:.2f}",
            'Number of Trades': n_trades,
            'Final Equity': f"${equity.iloc[-1]:,.2f}"
        }
        
        return metrics
    
    def plot_results(self, results: pd.DataFrame, title: str = "Backtest Results"):
        """Plot backtest results."""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Equity curve
        ax1 = axes[0]
        ax1.plot(results.index, results['equity'], label='Strategy', linewidth=2)
        ax1.plot(results.index, self.initial_capital * results['cumulative_returns'], 
                label='Buy & Hold', linewidth=2, alpha=0.7)
        ax1.set_ylabel('Equity ($)')
        ax1.set_title(f'{title} - Equity Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Drawdown
        ax2 = axes[1]
        equity = results['equity']
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max * 100
        ax2.fill_between(results.index, drawdown, 0, color='red', alpha=0.3)
        ax2.plot(results.index, drawdown, color='red', linewidth=1)
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_title('Drawdown')
        ax2.grid(True, alpha=0.3)
        
        # Positions
        ax3 = axes[2]
        ax3.plot(results.index, results['position'], linewidth=1)
        ax3.set_ylabel('Position')
        ax3.set_title('Trading Positions')
        ax3.set_ylim([-1.5, 1.5])
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

def walk_forward_backtest(signals: pd.DataFrame, 
                         data: pd.DataFrame,
                         cfg: dict = None) -> Dict:
    """
    Perform walk-forward backtesting.
    """
    if cfg is None:
        # Default configuration
        train_days = 4
        test_days = 3
    else:
        train_days = cfg['split']['train_days']
        test_days = cfg['split']['test_days']
    
    # Initialize backtester
    bt = Backtester()
    
    # Simple backtest for now (can be extended to walk-forward)
    results = bt.calculate_returns(signals, data)
    metrics = bt.calculate_metrics(results)
    
    # Create summary DataFrame
    metrics_df = pd.DataFrame([metrics]).T
    metrics_df.columns = ['Value']
    
    # Plot results
    fig = bt.plot_results(results)
    
    return {
        'results': results,
        'metrics': metrics_df,
        'equity_curve': results[['equity', 'cumulative_returns']],
        'figure': fig
    }