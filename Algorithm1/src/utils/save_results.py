import pandas as pd
import numpy as np
from typing import Dict
import logging
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

logger = logging.getLogger(__name__)

def save_results(results: Dict, config: Dict):
    """Save backtest results and generate plots."""
    logger.info("Saving results...")
    
    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save results to JSON
    results_file = results_dir / f"backtest_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {results_file}")
    
    # Generate and save plots
    plot_portfolio_value(results, results_dir, timestamp)
    plot_model_performance(results, results_dir, timestamp)
    plot_rule_performance(results, results_dir, timestamp)
    
    logger.info("Results saved successfully")

def plot_portfolio_value(results: Dict, results_dir: Path, timestamp: str):
    """Plot portfolio value over time."""
    plt.figure(figsize=(12, 6))
    
    # Plot portfolio value
    portfolio_values = results['overall']['portfolio_values']
    plt.plot(portfolio_values, label='Portfolio Value')
    
    # Add horizontal line for initial capital
    plt.axhline(y=results['overall']['initial_capital'], 
                color='r', linestyle='--', 
                label='Initial Capital')
    
    # Customize plot
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Time')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.savefig(results_dir / f"portfolio_value_{timestamp}.png")
    plt.close()

def plot_model_performance(results: Dict, results_dir: Path, timestamp: str):
    """Plot model performance metrics."""
    plt.figure(figsize=(12, 6))
    
    # Prepare data
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    model_names = list(results['models'].keys())
    
    # Create grouped bar plot
    x = np.arange(len(model_names))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = [results['models'][model][metric] for model in model_names]
        plt.bar(x + i*width, values, width, label=metric)
    
    # Customize plot
    plt.title('Model Performance Comparison')
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.xticks(x + width*1.5, model_names)
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.savefig(results_dir / f"model_performance_{timestamp}.png")
    plt.close()

def plot_rule_performance(results: Dict, results_dir: Path, timestamp: str):
    """Plot top rule performance."""
    if not results['rules']:
        return
        
    plt.figure(figsize=(12, 6))
    
    # Get top 10 rules
    top_rules = results['rules'][:10]
    
    # Prepare data
    rule_names = [f"Rule {i+1}" for i in range(len(top_rules))]
    metrics = ['precision', 'recall', 'f1_score']
    
    # Create grouped bar plot
    x = np.arange(len(rule_names))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        values = [rule[metric] for rule in top_rules]
        plt.bar(x + i*width, values, width, label=metric)
    
    # Customize plot
    plt.title('Top 10 Rules Performance')
    plt.xlabel('Rules')
    plt.ylabel('Score')
    plt.xticks(x + width, rule_names)
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.savefig(results_dir / f"rule_performance_{timestamp}.png")
    plt.close() 