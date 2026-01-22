import streamlit as st
import os
from pathlib import Path
from datetime import datetime
import sys

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag import init_rag_system
from src.report_generator import ClimateReportGenerator

# Page config
st.set_page_config(
    page_title="Algeria Climate AI",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #212121;
    }
    
    .main .block-container {
        max-width: 800px;
        padding-top: 3rem;
        padding-bottom: 5rem;
    }
    
    .main-title {
        color: #ffffff;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 500;
        margin-bottom: 2rem;
    }
    
    .stChatInput > div {
        background-color: #303030;
        border-radius: 24px;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #171717;
    }
    
    p, span, label {
        color: #ececec;
    }
    
    .welcome-text {
        color: #ececec;
        text-align: center;
        opacity: 0.8;
        font-size: 0.9rem;
        margin-top: 1rem;
    }
    
    .feedback-btn {
        background: transparent;
        border: 1px solid #424242;
        border-radius: 8px;
        padding: 4px 12px;
        margin: 2px;
        cursor: pointer;
    }
    
    .feedback-btn:hover {
        background: #303030;
    }
</style>
""", unsafe_allow_html=True)

# Get API key from environment
groq_api_key = os.getenv("GROQ_API_KEY", "")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_system" not in st.session_state:
    st.session_state.rag_system = None

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

if "feedback" not in st.session_state:
    st.session_state.feedback = {}  # {message_idx: "up" or "down"}

# Initialize RAG system
if groq_api_key and st.session_state.rag_system is None:
    try:
        with st.spinner("Loading AI system..."):
            st.session_state.rag_system = init_rag_system(groq_api_key)
    except Exception as e:
        st.error(f"Failed to initialize: {e}")

# --- SIDEBAR (Moved to top for consistent rendering) ---
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    if groq_api_key and st.session_state.rag_system:
        st.success(f"✅ Connected ({st.session_state.rag_system.collection.count()} docs)")
    elif groq_api_key:
        st.warning("⏳ Initializing...")
    else:
        st.error("❌ No API key")
    
    st.markdown("---")
    
    # Chat controls
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.feedback = {}
        st.rerun()
    
    if st.button("🔄 Reload DB", use_container_width=True):
        if groq_api_key:
            st.session_state.rag_system = init_rag_system(groq_api_key, reset_db=True)
            st.success("✅ Reloaded!")
            st.rerun()
            
    st.markdown("---")
    st.markdown("### 📄 Reports")
    
    report_path = "climate_report.pdf"
    if st.button("📊 Generate PDF Report", use_container_width=True):
        with st.spinner("Generating detailed report (this may take a moment)..."):
            try:
                generator = ClimateReportGenerator()
                generator.generate_report(report_path)
                st.success("Report generated!")
            except Exception as e:
                st.error(f"Failed to generate: {e}")
    
    if os.path.exists(report_path):
        with open(report_path, "rb") as f:
            st.download_button(
                "⬇️ Download Report",
                data=f,
                file_name="Algeria_Climate_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )

# --- QUERY PROCESSING ---
def process_streaming_query(query: str):
    """Process query with streaming response"""
    st.session_state.messages.append({"role": "user", "content": query})

# Check for pending query
if st.session_state.pending_query:
    process_streaming_query(st.session_state.pending_query)
    st.session_state.pending_query = None

# --- MAIN CHAT INTERFACE ---
if not st.session_state.messages:
    # Welcome screen
    st.markdown('<h1 class="main-title">🇩🇿 What can I help you understand about Algeria\'s climate?</h1>', unsafe_allow_html=True)
    
    suggestions = [
        "What are the temperature trends?",
        "Is there evidence of drought?",
        "What do forecasts predict for 2040?",
        "Summarize the climate analysis"
    ]
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(suggestions[0], key="s1", use_container_width=True):
            st.session_state.pending_query = suggestions[0]
            st.rerun()
        if st.button(suggestions[2], key="s3", use_container_width=True):
            st.session_state.pending_query = suggestions[2]
            st.rerun()
    
    with col2:
        if st.button(suggestions[1], key="s2", use_container_width=True):
            st.session_state.pending_query = suggestions[1]
            st.rerun()
        if st.button(suggestions[3], key="s4", use_container_width=True):
            st.session_state.pending_query = suggestions[3]
            st.rerun()
    
    if not groq_api_key:
        st.markdown('<p class="welcome-text">⚠️ Set GROQ_API_KEY in .env file</p>', unsafe_allow_html=True)
    elif st.session_state.rag_system:
        doc_count = st.session_state.rag_system.collection.count()
        st.markdown(f'<p class="welcome-text">✅ Ready ({doc_count} documents indexed)</p>', unsafe_allow_html=True)

else:
    # Chat history with feedback buttons
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Add feedback buttons for assistant messages
            if message["role"] == "assistant":
                col1, col2, col3 = st.columns([1, 1, 10])
                
                current_feedback = st.session_state.feedback.get(i)
                
                with col1:
                    if st.button("👍", key=f"up_{i}", 
                                 type="primary" if current_feedback == "up" else "secondary"):
                        st.session_state.feedback[i] = "up"
                        st.rerun()
                
                with col2:
                    if st.button("👎", key=f"down_{i}",
                                 type="primary" if current_feedback == "down" else "secondary"):
                        st.session_state.feedback[i] = "down"
                        st.rerun()
    
    # Check if last message needs AI response (streaming)
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        with st.chat_message("assistant"):
            if st.session_state.rag_system:
                # Stream the response
                response_placeholder = st.empty()
                full_response = ""
                
                for chunk in st.session_state.rag_system.query_stream(
                    st.session_state.messages[-1]["content"]
                ):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                
                response_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                st.rerun()
            else:
                msg = "Please set GROQ_API_KEY in .env file."
                st.warning(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})

# Chat input
if prompt := st.chat_input("Ask about Algeria's climate..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()
