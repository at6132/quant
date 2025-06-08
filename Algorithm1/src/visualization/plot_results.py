import pandas as pd
import numpy as np
from typing import Dict
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)

def plot_pnl_curves(results: Dict, output_dir: str):
    """Plot PnL curves for each model."""
    plt.figure(figsize=(15, 10))
    
    # Plot each model type
    for model_type, model_results in results.items():
        for signal, metrics in model_results.items():
            # Create cumulative PnL curve
            pnl = pd.Series(metrics['pnl'])
            cum_pnl = pnl.cumsum()
            
            # Plot
            plt.plot(cum_pnl.index, cum_pnl.values, label=signal)
    
    plt.title('Cumulative PnL Curves')
    plt.xlabel('Time')
    plt.ylabel('Cumulative PnL ($)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f"{output_dir}/pnl_curves.png")
    plt.close()

def plot_metrics_heatmap(results: Dict, output_dir: str):
    """Plot metrics heatmap for each model."""
    # Create metrics DataFrame
    metrics_df = pd.DataFrame()
    
    for model_type, model_results in results.items():
        for signal, metrics in model_results.items():
            metrics_df.loc[signal, 'Total Return'] = metrics['total_return']
            metrics_df.loc[signal, 'Sharpe'] = metrics['sharpe']
            metrics_df.loc[signal, 'Win Rate'] = metrics['win_rate']
            metrics_df.loc[signal, 'Profit Factor'] = metrics['profit_factor']
            metrics_df.loc[signal, 'Max Drawdown'] = metrics['max_drawdown']
    
    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(metrics_df, annot=True, fmt='.2f', cmap='RdYlGn', center=0)
    plt.title('Model Performance Metrics')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f"{output_dir}/metrics_heatmap.png")
    plt.close()

def plot_drawdown_curves(results: Dict, output_dir: str):
    """Plot drawdown curves for each model."""
    plt.figure(figsize=(15, 10))
    
    for model_type, model_results in results.items():
        for signal, metrics in model_results.items():
            # Calculate drawdown
            pnl = pd.Series(metrics['pnl'])
            cum_pnl = pnl.cumsum()
            drawdown = cum_pnl - cum_pnl.cummax()
            
            # Plot
            plt.plot(drawdown.index, drawdown.values, label=signal)
    
    plt.title('Drawdown Curves')
    plt.xlabel('Time')
    plt.ylabel('Drawdown ($)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f"{output_dir}/drawdown_curves.png")
    plt.close()

def plot_results(results: Dict, output_dir: str):
    """
    Generate and save plots for backtest results.
    
    Args:
        results: Dictionary with backtest results
        output_dir: Directory to save plots
    """
    logger.info("Generating plots...")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    plot_pnl_curves(results, output_dir)
    plot_metrics_heatmap(results, output_dir)
    plot_drawdown_curves(results, output_dir)
    
    logger.info(f"Plots saved to {output_dir}/") 