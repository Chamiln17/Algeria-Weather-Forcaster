import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag import init_rag_system

st.set_page_config(page_title="Algeria Climate Analysis", layout="wide")

st.title("🇩🇿 Algeria Climate Change Analysis & Reporting")

# Sidebar
st.sidebar.header("Configuration")
groq_api_key = st.sidebar.text_input(
    "Groq API Key", 
    type="password",
    help="Get your free API key from https://console.groq.com/keys"
)

if not groq_api_key:
    st.warning("⚠️ Please enter your Groq API Key in the sidebar to use AI features.")
    st.info("💡 Get a free API key (14,400 requests/day) at https://console.groq.com/keys")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🤖 AI Report", "📁 Data Explorer"])

# Tab 1: Dashboard
with tab1:
    st.header("Climate Trends Dashboard")
    
    results_dir = Path("Results")
    if results_dir.exists():
        images = list(results_dir.glob("*.png"))
        if images:
            selected_img = st.selectbox("Select Visualization", [img.name for img in images])
            st.image(str(results_dir / selected_img), caption=selected_img, width=800)
        else:
            st.info("No visualizations found in Results/ directory.")

# Tab 2: AI Report Generator
with tab2:
    st.header("AI Climate Report Generator")
    st.markdown("**Powered by:** ChromaDB (local) + e5-small embeddings + Groq Llama 3.1 8B")
    
    if groq_api_key:
        # Initialize RAG system
        if 'rag_system' not in st.session_state:
            with st.spinner("🔄 Initializing RAG system..."):
                try:
                    st.session_state.rag_system = init_rag_system(groq_api_key)
                    st.success(f"✅ RAG Ready! ({st.session_state.rag_system.collection.count()} documents loaded)")
                except FileNotFoundError as e:
                    st.error(f"❌ {str(e)}")
                    st.info("Run: `python src/generate_stats_db.py` first")
                    st.stop()
                except Exception as e:
                    st.error(f"❌ Failed to initialize: {e}")
                    st.stop()
        
        # Reinitialize button
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🔄 Reinitialize DB"):
                with st.spinner("Reinitializing..."):
                    st.session_state.rag_system = init_rag_system(groq_api_key, reset_db=True)
                    st.success("✅ Database reinitialized!")
        
        # Query interface
        query = st.text_area(
            "Ask a question about Algeria's climate:",
            "Summarize the key temperature trends and future risks for Algeria based on the analysis.",
            height=100
        )
        
        if st.button("🚀 Generate Report", type="primary"):
            if 'rag_system' in st.session_state:
                with st.spinner("🤖 Analyzing data..."):
                    try:
                        response = st.session_state.rag_system.query(query)
                        st.markdown("### 📝 Analysis Report")
                        st.markdown(response)
                        
                        # Show quota info
                        st.info("💡 Groq free tier: 14,400 requests/day remaining")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
            else:
                st.error("RAG system not initialized")
    else:
        st.info("👆 Enter your Groq API key in the sidebar to enable AI reporting")

# Tab 3: Data Explorer
with tab3:
    st.header("Data Explorer")
    
    files = list(Path("Predictions").glob("*.csv")) + list(Path("Preprocessed_dataset").glob("*.csv"))
    
    if files:
        selected_file = st.selectbox("Select File", [f.name for f in files])
        file_path = next(f for f in files if f.name == selected_file)
        
        try:
            df = pd.read_csv(file_path)
            st.dataframe(df.head(100), use_container_width=True)
            st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        st.info("No data files found")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 System Info")
if 'rag_system' in st.session_state:
    st.sidebar.success(f"✅ {st.session_state.rag_system.collection.count()} docs indexed")
else:
    st.sidebar.info("⏳ RAG not initialized")
