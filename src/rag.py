"""
RAG System for Climate Reporting (Powered by Gemini)
Lightweight version - no embeddings to avoid quota issues
"""
import os
import json
import logging
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

# Constants
RESULTS_DIR = Path('Results')
PREDICTIONS_DIR = Path('Predictions')

def load_text_data() -> str:
    """Load and format data for RAG context"""
    context = []
    
    # 1. Load Trends
    trends_path = RESULTS_DIR / 'trends.json'
    if trends_path.exists():
        try:
            with open(trends_path, 'r') as f:
                trends = json.load(f)
            context.append("## Historical Trends Analysis")
            context.append(json.dumps(trends, indent=2))
        except Exception as e:
            logger.error(f"Error loading trends: {e}")
    
    # 2. Load Forecasts
    context.append("\n## Future Forecasts (2025-2040)")
    for file in PREDICTIONS_DIR.glob('*_forecast_2040*.csv'):
        try:
            df = pd.read_csv(file)
            summary = df.describe().to_string()
            context.append(f"### Model: {file.stem}")
            context.append(summary)
            context.append("Last 5 years prediction (sample):")
            context.append(df.tail(12).to_string()) 
        except Exception as e:
            logger.error(f"Error loading forecast {file}: {e}")

    return "\n".join(context)

def init_rag_index(api_key: str):
    """Initialize RAG with Gemini - lightweight version"""
    if not api_key:
        raise ValueError("Gemini API Key is required")
        
    os.environ['GOOGLE_API_KEY'] = api_key
    
    # Load all context data
    text_data = load_text_data()
    if not text_data.strip():
        text_data = "No data available for analysis. Please ensure trends and forecasts are generated."
    
    # Return the context directly (no indexing needed)
    return {"context": text_data, "api_key": api_key}

def generate_climate_report(query: str, index_data: dict) -> str:
    """Query the RAG system using direct Gemini API call"""
    try:
        import google.generativeai as genai
        
        genai.configure(api_key=index_data["api_key"])
        model = genai.GenerativeModel('models/gemini-2.0-flash')
        
        # Create prompt with context
        prompt = f"""You are a climate data analyst. Use the following data to answer the question.

DATA:
{index_data["context"]}

QUESTION: {query}

Provide a detailed, data-driven answer based on the information above."""
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return f"Error: {str(e)}"
