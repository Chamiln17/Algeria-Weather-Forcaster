"""
PDF Report Generator for Climate Analysis

Generates a detailed PDF report including:
- Executive Summary
- Forecast Charts (Temperature & ET0)
- Statistical Analysis
- Model Performance
"""

from fpdf import FPDF
from datetime import datetime
from pathlib import Path
import pandas as pd
import json
import sys
from pathlib import Path

# Add project root to path to ensure imports work
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.visualizer import ClimateVisualizer

class ClimateReportGenerator(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        self.visualizer = ClimateVisualizer(
            'RL Agent/final_rl_temperature_forecast_2040_with_uncertainty.csv',
            'RL Agent/final_rl_et0_forecast_2040_with_uncertainty.csv'
        )
        self.stats_db = self._load_stats_db()

    def _load_stats_db(self):
        try:
            with open('Results/stats_db.json', 'r') as f:
                return json.load(f)
        except Exception as e:
            return {}

    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Algeria Climate Forecast Report 2024-2040', 0, 1, 'C')
        self.set_font('Arial', 'I', 10)
        self.cell(0, 10, f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, title, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, text):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 6, text)
        self.ln()

    def add_plot(self, plot_path, caption):
        if plot_path and Path(plot_path).exists():
            # Get available width
            w = self.w - 2 * self.l_margin
            self.image(plot_path, w=w)
            self.set_font('Arial', 'I', 9)
            self.cell(0, 5, caption, 0, 1, 'C')
            self.ln(10)

    def generate_report(self, output_path='climate_report.pdf'):
        self.add_page()
        
        # 1. Executive Summary
        self.chapter_title('Executive Summary')
        summary_text = (
            "This report presents a comprehensive analysis of climate trends in Algeria for the period 2024-2040. "
            "Forecasts are generated using a Reinforcement Learning (RL) agent that dynamically selects "
            "the best performing model (LSTM or SARIMA) for each month. "
            "The analysis covers two key variables: Temperature (2m mean) and Reference Evapotranspiration (ET0)."
        )
        self.chapter_body(summary_text)

        # 2. Temperature Forecast
        self.chapter_title('Temperature Forecast (2024-2040)')
        
        # Plot
        plot_path = self.visualizer.generate_forecast_plot('temperature')
        self.add_plot(plot_path, 'Figure 1: Temperature Forecast with 95% Confidence Interval')
        
        # Stats
        if 'RL_Agent_Temperature' in self.stats_db['forecasts']:
            stats = self.stats_db['forecasts']['RL_Agent_Temperature']['summary_statistics']['RL_Best_Forecast']
            self.set_font('Arial', '', 10)
            self.cell(0, 6, f"Mean Forecast: {stats['mean']:.2f} deg C", 0, 1)
            self.cell(0, 6, f"Maximum Predicted: {stats['max']:.2f} deg C", 0, 1)
            self.cell(0, 6, f"Minimum Predicted: {stats['min']:.2f} deg C", 0, 1)
            self.ln(5)

        # 3. ET0 Forecast
        self.add_page()
        self.chapter_title('Evapotranspiration (ET0) Forecast (2024-2040)')
        
        # Plot
        plot_path = self.visualizer.generate_forecast_plot('et0')
        self.add_plot(plot_path, 'Figure 2: ET0 Forecast with 95% Confidence Interval')
        
        # Stats
        if 'RL_Agent_ET0' in self.stats_db['forecasts']:
            stats = self.stats_db['forecasts']['RL_Agent_ET0']['summary_statistics']['RL_Best_Forecast']
            self.set_font('Arial', '', 10)
            self.cell(0, 6, f"Mean Forecast: {stats['mean']:.2f} mm", 0, 1)
            self.cell(0, 6, f"Maximum Predicted: {stats['max']:.2f} mm", 0, 1)
            self.cell(0, 6, f"Minimum Predicted: {stats['min']:.2f} mm", 0, 1)
            self.ln(5)

        # 4. Model Analysis
        self.add_page()
        self.chapter_title('Model Performance & Uncertainty')
        
        # Uncertainty Plot
        plot_path = self.visualizer.generate_uncertainty_plot('temperature')
        self.add_plot(plot_path, 'Figure 3: Temperature Forecast Uncertainty Growth')
        
        model_text = (
            "The RL Agent utilizes an ensemble of models including LSTM, SARIMA, Ridge, and Prophet. "
            "For Temperature, LSTM is typically selected for its ability to capture complex non-linear patterns, "
            "while SARIMA is favored for stable seasonal transitions. "
            "Uncertainty quantification incorporates both empirical backcast errors and ensemble spread."
        )
        self.chapter_body(model_text)

        # Output
        self.output(output_path)
        return output_path

if __name__ == '__main__':
    generator = ClimateReportGenerator()
    path = generator.generate_report()
    print(f"Report generated: {path}")
