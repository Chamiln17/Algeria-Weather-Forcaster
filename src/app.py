import streamlit as st
import os
from pathlib import Path
import sys

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag import init_rag_system

# Page config
st.set_page_config(
    page_title="Algeria Climate AI",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for ChatGPT-like dark theme
st.markdown("""
<style>
    .stApp {
        background-color: #212121;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .main .block-container {
        max-width: 800px;
        padding-top: 5rem;
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

# Initialize RAG system if API key exists (only once)
if groq_api_key and st.session_state.rag_system is None:
    try:
        with st.spinner("Loading AI system..."):
            st.session_state.rag_system = init_rag_system(groq_api_key)
    except Exception as e:
        st.error(f"Failed to initialize: {e}")

# Function to process a query
def process_query(query: str):
    """Add user message and get AI response"""
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Get AI response
    if st.session_state.rag_system:
        try:
            response = st.session_state.rag_system.query(query)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
    else:
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "Please set GROQ_API_KEY in .env file to enable AI responses."
        })

# Check if there's a pending query from suggestion buttons
if st.session_state.pending_query:
    process_query(st.session_state.pending_query)
    st.session_state.pending_query = None

# Main chat interface
if not st.session_state.messages:
    # Welcome screen
    st.markdown('<h1 class="main-title">🇩🇿 What can I help you understand about Algeria\'s climate?</h1>', unsafe_allow_html=True)
    
    # Suggestion chips
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
        st.markdown('<p class="welcome-text">⚠️ Set GROQ_API_KEY in .env file to enable AI features</p>', unsafe_allow_html=True)
    elif st.session_state.rag_system:
        doc_count = st.session_state.rag_system.collection.count()
        st.markdown(f'<p class="welcome-text">✅ Ready ({doc_count} documents indexed)</p>', unsafe_allow_html=True)

else:
    # Chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about Algeria's climate..."):
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add to history and get response
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate response
    with st.chat_message("assistant"):
        if st.session_state.rag_system:
            with st.spinner("Analyzing..."):
                try:
                    response = st.session_state.rag_system.query(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
        else:
            msg = "Please set your GROQ_API_KEY in the .env file."
            st.warning(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})

# Minimal sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    if groq_api_key and st.session_state.rag_system:
        st.success(f"✅ Connected ({st.session_state.rag_system.collection.count()} docs)")
    elif groq_api_key:
        st.warning("⏳ Initializing...")
    else:
        st.error("❌ No API key")
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    if st.button("🔄 Reinitialize DB", use_container_width=True):
        if groq_api_key:
            st.session_state.rag_system = init_rag_system(groq_api_key, reset_db=True)
            st.success("✅ Reinitialized!")
            st.rerun()
    
    st.markdown("---")
    st.caption("Powered by Groq + ChromaDB")
