import streamlit as st
import pandas as pd
import json
import os
import sys
from pathlib import Path

# Add src to pythonpath
sys.path.append(str(Path(__file__).parent.parent))

from src.rag import init_rag_index, generate_climate_report
from src.utils import load_csv_with_dates

st.set_page_config(page_title="Algeria Climate Analysis", layout="wide")

st.title("🇩🇿 Algeria Climate Change Analysis & Reporting")

# Sidebar for Config
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Google Gemini API Key", type="password")

if not api_key:
    st.warning("Please enter your Google Gemini API Key in the sidebar to use the RAG features.")

# Main Interface
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🤖 AI Report", "📁 Data Explorer"])

with tab1:
    st.header("Climate Trends Dashboard")
    
    # Load and display images from Results folder
    results_dir = Path("Results")
    if results_dir.exists():
        images = list(results_dir.glob("*.png"))
        if images:
            selected_img = st.selectbox("Select Visualization", [img.name for img in images])
            st.image(str(results_dir / selected_img), caption=selected_img, use_column_width=True)
        else:
            st.info("No visualizations found in Results/ directory.")

with tab2:
    st.header("AI Climate Report Generator")
    
    if api_key:
        if 'rag_index' not in st.session_state:
            with st.spinner("Initializing AI Knowledge Base..."):
                try:
                    st.session_state.rag_index = init_rag_index(api_key)
                    st.success("AI Ready!")
                except Exception as e:
                    st.error(f"Failed to initialize AI: {e}")
        
        query = st.text_area("Ask a question about the climate data:", 
                           "Summarize the key temperature trends and future risks for Algeria based on the analysis.")
        
        if st.button("Generate Report"):
            if 'rag_index' in st.session_state:
                with st.spinner("Analyzing data..."):
                    response = generate_climate_report(query, st.session_state.rag_index)
                    st.markdown("### 📝 Analysis Report")
                    st.markdown(response)
            else:
                st.error("AI Index not initialized.")
    else:
        st.info("Enter API Key to enable AI reporting.")

with tab3:
    st.header("Data Explorer")
    # List available CSVs
    files = list(Path("Predictions").glob("*.csv")) + list(Path("Preprocessed_dataset").glob("*.csv"))
    
    if files:
        selected_file = st.selectbox("Select File", [f.name for f in files])
        file_path = next(f for f in files if f.name == selected_file)
        
        try:
            df = pd.read_csv(file_path)
            st.dataframe(df.head(100))
            st.write(f"Shape: {df.shape}")
        except Exception as e:
            st.error(f"Error reading file: {e}")

