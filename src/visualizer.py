"""
Visualization Generator for Climate RAG System

Automatically generates charts based on query intent:
- Temperature/ET0 forecasts with confidence intervals
- Model comparison plots
- Historical trends
- Uncertainty visualization
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ClimateVisualizer:
    """Generate visualizations for climate data"""
    
    def __init__(self, temp_forecast_path, et0_forecast_path):
        """
        Initialize with forecast data paths
        
        Args:
            temp_forecast_path: Path to temperature forecast CSV
            et0_forecast_path: Path to ET0 forecast CSV
        """
        self.temp_df = pd.read_csv(temp_forecast_path)
        self.temp_df['Date'] = pd.to_datetime(self.temp_df['Date'])
        
        self.et0_df = pd.read_csv(et0_forecast_path)
        self.et0_df['Date'] = pd.to_datetime(self.et0_df['Date'])
        
        # Check if uncertainty columns exist
        self.has_uncertainty = 'RL_Best_Forecast_Lower' in self.temp_df.columns
    
    def detect_viz_intent(self, query: str) -> dict:
        """
        Detect if query requires visualization
        
        Returns:
            dict: {'needs_viz': bool, 'viz_type': str, 'variable': str}
        """
        query_lower = query.lower()
        
        # Visualization trigger words
        viz_triggers = ['plot', 'graph', 'chart', 'show', 'visualize', 'display']
        needs_viz = any(trigger in query_lower for trigger in viz_triggers)
        
        # Also trigger on forecast questions
        if 'forecast' in query_lower or '2040' in query_lower or '2030' in query_lower:
            needs_viz = True
        
        # Detect variable
        if 'et0' in query_lower or 'evapotranspiration' in query_lower:
            variable = 'et0'
        elif 'temp' in query_lower:
            variable = 'temperature'
        else:
            variable = 'temperature'  # Default
        
        # Detect viz type
        if 'model' in query_lower and ('comparison' in query_lower or 'compare' in query_lower):
            viz_type = 'model_comparison'
        elif 'uncertainty' in query_lower or 'confidence' in query_lower:
            viz_type = 'uncertainty'
        else:
            viz_type = 'forecast'  # Default
        
        return {
            'needs_viz': needs_viz,
            'viz_type': viz_type,
            'variable': variable
        }
    
    def generate_forecast_plot(self, variable: str, year_range: tuple = None) -> str:
        """
        Generate forecast plot with confidence intervals
        
        Args:
            variable: 'temperature' or 'et0'
            year_range: Optional (start_year, end_year) tuple
        
        Returns:
            str: Path to saved image
        """
        df = self.temp_df if variable == 'temperature' else self.et0_df
        unit = '°C' if variable == 'temperature' else 'mm'
        title = f"{variable.upper()} Forecast 2024-2040 (RL Agent)"
        
        # Filter by year range if specified
        if year_range:
            df = df[(df['Date'].dt.year >= year_range[0]) & 
                   (df['Date'].dt.year <= year_range[1])]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot main forecast
        ax.plot(df['Date'], df['RL_Best_Forecast'], 
               linewidth=2.5, label='RL Agent Forecast', color='#1f77b4')
        
        # Add confidence interval if available
        if self.has_uncertainty:
            ax.fill_between(df['Date'], 
                           df['RL_Best_Forecast_Lower'], 
                           df['RL_Best_Forecast_Upper'],
                           alpha=0.3, color='#1f77b4', label='95% Confidence Interval')
        
        # Formatting
        ax.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{variable.title()} ({unit})', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save
        output_path = Path('temp_viz') / f'{variable}_forecast_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        output_path.parent.mkdir(exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def generate_model_comparison(self, variable: str) -> str:
        """
        Generate plot comparing all models
        
        Args:
            variable: 'temperature' or 'et0'
        
        Returns:
            str: Path to saved image
        """
        df = self.temp_df if variable == 'temperature' else self.et0_df
        unit = '°C' if variable == 'temperature' else 'mm'
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot each model
        models = ['SARIMA', 'LSTM', 'Ridge', 'Prophet']
        colors = ['#2ca02c', '#ff7f0e', '#d62728', '#9467bd']
        
        for model, color in zip(models, colors):
            if model in df.columns:
                # Thin lines for individual models
                ax.plot(df['Date'], df[model], linewidth=1.5, alpha=0.6,
                       label=model, color=color)
        
        # Bold line for RL selection
        ax.plot(df['Date'], df['RL_Best_Forecast'], 
               linewidth=3, label='RL Agent (Best)', color='#1f77b4',
               linestyle='--')
        
        # Formatting
        ax.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{variable.title()} ({unit})', fontsize=12, fontweight='bold')
        ax.set_title(f'Model Comparison: {variable.upper()}', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=10, ncol=2)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save
        output_path = Path('temp_viz') / f'{variable}_models_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        output_path.parent.mkdir(exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def generate_uncertainty_plot(self, variable: str) -> str:
        """
        Generate plot showing uncertainty growth over time
        
        Args:
            variable: 'temperature' or 'et0'
        
        Returns:
            str: Path to saved image
        """
        if not self.has_uncertainty:
            return None
        
        df = self.temp_df if variable == 'temperature' else self.et0_df
        unit = '°C' if variable == 'temperature' else 'mm'
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Top plot: Forecast with uncertainty bands
        ax1.plot(df['Date'], df['RL_Best_Forecast'], 
                linewidth=2, label='Forecast', color='#1f77b4')
        ax1.fill_between(df['Date'], 
                        df['RL_Best_Forecast_Lower'], 
                        df['RL_Best_Forecast_Upper'],
                        alpha=0.3, color='#1f77b4', label='95% CI')
        
        ax1.set_ylabel(f'{variable.title()} ({unit})', fontsize=12, fontweight='bold')
        ax1.set_title(f'{variable.upper()} Forecast with Uncertainty', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Bottom plot: Uncertainty growth
        ax2.plot(df['Date'], df['RL_Best_Forecast_Std'], 
                linewidth=2, color='#ff7f0e', label='Uncertainty (±1σ)')
        ax2.fill_between(df['Date'], 0, df['RL_Best_Forecast_Std'],
                        alpha=0.3, color='#ff7f0e')
        
        ax2.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax2.set_ylabel(f'Uncertainty ({unit})', fontsize=12, fontweight='bold')
        ax2.set_title('Forecast Uncertainty Growth', fontsize=14, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis for both
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save
        output_path = Path('temp_viz') / f'{variable}_uncertainty_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        output_path.parent.mkdir(exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def generate_visualization(self, query: str) -> tuple:
        """
        Main entry point: detect intent and generate appropriate visualization
        
        Args:
            query: User's query string
        
        Returns:
            tuple: (viz_path: str or None, viz_description: str)
        """
        intent = self.detect_viz_intent(query)
        
        if not intent['needs_viz']:
            return None, None
        
        variable = intent['variable']
        viz_type = intent['viz_type']
        
        try:
            if viz_type == 'model_comparison':
                path = self.generate_model_comparison(variable)
                desc = f"Model comparison for {variable}"
            elif viz_type == 'uncertainty':
                path = self.generate_uncertainty_plot(variable)
                desc = f"Uncertainty visualization for {variable}"
            else:  # Default: forecast
                path = self.generate_forecast_plot(variable)
                desc = f"Forecast plot for {variable} with 95% confidence interval"
            
            return path, desc
        
        except Exception as e:
            print(f"Error generating visualization: {e}")
            return None, None

if __name__ == "__main__":
    # Test visualization generation
    visualizer = ClimateVisualizer(
        'RL Agent/final_rl_temperature_forecast_2040_with_uncertainty.csv',
        'RL Agent/final_rl_et0_forecast_2040_with_uncertainty.csv'
    )
    
    # Test queries
    test_queries = [
        "Show me temperature forecast for 2040",
        "Compare all models for ET0",
        "What's the forecast uncertainty?",
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        path, desc = visualizer.generate_visualization(query)
        if path:
            print(f"✅ Generated: {desc}")
            print(f"   Saved to: {path}")
        else:
            print("❌ No visualization needed")
