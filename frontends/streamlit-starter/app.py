"""
IONOS Chatbot Streamlit Application

This application provides a web interface for interacting with the IONOS chatbot backend.
Features include:
- RAG (Retrieval-Augmented Generation) initialization from web URLs
- Model selection for different LLaMA variants
- Interactive chat interface with message history
- Real-time communication with FastAPI backend
"""
import os

import streamlit as st
import requests
from dotenv import load_dotenv

# Load env (.env in project root or current dir)
load_dotenv()

# Backend API configuration (local dev)
BACKEND_URL = "http://127.0.0.1:8000"

# Normalize model id to ASCII-safe (HTTP headers are latin-1)
def normalize_model_id(mid: str) -> str:
    # Replace common unicode dashes with ASCII hyphen
    return (
        (mid or "")
        .replace("\u2010", "-")  # hyphen
        .replace("\u2011", "-")  # non-breaking hyphen
        .replace("\u2012", "-")  # figure dash
        .replace("\u2013", "-")  # en dash
        .replace("\u2014", "-")  # em dash
        .strip()
    )

# Custom CSS for beautiful styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
        border-right: 2px solid #3498db;
    }
    
    /* Beautiful chat bubbles */
    .user-bubble {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 25px 25px 5px 25px;
        margin: 10px 0 10px 20%;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        position: relative;
        animation: slideInRight 0.3s ease-out;
    }
    
    .ai-bubble {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 25px 25px 25px 5px;
        margin: 10px 20% 10px 0;
        box-shadow: 0 4px 15px rgba(240, 147, 251, 0.3);
        position: relative;
        animation: slideInLeft 0.3s ease-out;
    }
    
    /* Animations */
    @keyframes slideInRight {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideInLeft {
        from { transform: translateX(-100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #e0e0e0;
        padding: 15px 20px;
        font-size: 16px;
        transition: all 0.3s ease;
        background: rgba(255, 255, 255, 0.95);
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.3);
        transform: scale(1.02);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 12px 30px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        border: none;
        color: white;
    }
    
    /* Title styling */
    .main-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Sidebar title styling */
    .sidebar-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* Chat container */
    .chat-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Loading animation */
    .loading-dots {
        display: inline-block;
        position: relative;
        width: 80px;
        height: 80px;
    }
    
    .loading-dots div {
        position: absolute;
        top: 33px;
        width: 13px;
        height: 13px;
        border-radius: 50%;
        background: #667eea;
        animation-timing-function: cubic-bezier(0, 1, 1, 0);
    }
    
    .loading-dots div:nth-child(1) {
        left: 8px;
        animation: loading-dots1 0.6s infinite;
    }
    
    .loading-dots div:nth-child(2) {
        left: 8px;
        animation: loading-dots2 0.6s infinite;
    }
    
    .loading-dots div:nth-child(3) {
        left: 32px;
        animation: loading-dots2 0.6s infinite;
    }
    
    .loading-dots div:nth-child(4) {
        left: 56px;
        animation: loading-dots3 0.6s infinite;
    }
    
    @keyframes loading-dots1 {
        0% { transform: scale(0); }
        100% { transform: scale(1); }
    }
    
    @keyframes loading-dots3 {
        0% { transform: scale(1); }
        100% { transform: scale(0); }
    }
    
    @keyframes loading-dots2 {
        0% { transform: translate(0, 0); }
        100% { transform: translate(24px, 0); }
    }
    
    /* Success/Error messages */
    .success-message {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        color: white;
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        text-align: center;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(86, 171, 47, 0.3);
    }
    
    .error-message {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        text-align: center;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(255, 65, 108, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Configure Streamlit page settings
st.set_page_config(
    page_title="IONOS RAG Chatbot", 
    page_icon="✨", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Sidebar Configuration ---
with st.sidebar:
    st.markdown('<h1 class="sidebar-title">✨ RAG Chatbot</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # RAG Initialization Section
    st.markdown("### 🔗 Knowledge Base")
    st.markdown("*Initialize your AI with custom knowledge*")
    
    rag_url = st.text_input(
        "📄 Document URL", 
        value="", 
        placeholder="https://example.com",
        key="rag_url",
        help="Enter a webpage URL to load into the AI's knowledge base"
    )
    
    # Handle RAG initialization button click
    if st.button("🚀 Initialize Knowledge Base", key="init_rag_btn", use_container_width=True):
        if rag_url.strip():
            with st.spinner("🔍 Loading knowledge into AI..."):
                try:
                    resp = requests.post(f"{BACKEND_URL}/init", json={"page_url": rag_url})
                    if resp.ok:
                        st.markdown('<div class="success-message">✅ Knowledge base initialized successfully!</div>', unsafe_allow_html=True)
                        st.rerun()
                    else:
                        st.markdown('<div class="error-message">❌ Failed to initialize knowledge base</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f'<div class="error-message">❌ Error: {str(e)}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="error-message">⚠️ Please enter a valid URL</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Model Selection Section
    st.markdown("### 🤖 AI Model")
    st.markdown("*Choose your preferred AI model*")
    
    @st.cache_data(ttl=300)
    def fetch_model_ids():
        try:
            resp = requests.get(
                "https://openai.inference.de-txl.ionos.com/v1/models",
                headers={"Authorization": f"Bearer {os.getenv('IONOS_API_KEY')}"}
            )
            resp.raise_for_status()
            data = resp.json()
            return [m["id"] for m in data.get("data", [])]
        except Exception as e:
            st.error(f"Error fetching models: {e}")
            return []
    
    MODEL_OPTIONS = fetch_model_ids() or ["mistralai/Mistral-Small-24B-Instruct"]
    model = st.selectbox(
        "🎯 Select Model", 
        MODEL_OPTIONS, 
        key="model_select",
        help="Choose the AI model that will power your conversations"
    )
    
    # Ensure model is always available
    if not model:
        model = "mistralai/Mistral-Small-24B-Instruct"
    
    # Handle model switching - preserve chat history when model changes
    if "current_model" not in st.session_state:
        st.session_state["current_model"] = model
    elif st.session_state["current_model"] != model:
        st.session_state["current_model"] = model
        # Don't clear chat history - just update the model
        st.info(f"🤖 Switched to model: {model}")
    
    st.markdown("---")
    
    # Status section
    st.markdown("### 📊 Status")
    if "chat_history" in st.session_state:
        st.metric("💬 Messages", len(st.session_state["chat_history"]))
    else:
        st.metric("💬 Messages", 0)
    
    # Current model indicator
    st.markdown("### 🤖 Active Model")
    st.info(f"**{model}**")
    
    # Helpful tip
    st.markdown("💡 **Tip:** If a model is busy, simply switch to another model for immediate response!")

# --- Main Chat Interface ---
st.markdown('<h1 class="main-title">✨ IONOS RAG Chatbot</h1>', unsafe_allow_html=True)

# Initialize chat history
if "chat_history" not in st.session_state:
    try:
        resp = requests.get(f"{BACKEND_URL}/", headers={"x-model-id": normalize_model_id(model)})
        if resp.ok:
            st.session_state["chat_history"] = resp.json()
        else:
            st.session_state["chat_history"] = []
    except Exception:
        st.session_state["chat_history"] = []

# Display chat messages in beautiful bubbles
if st.session_state["chat_history"]:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    for msg in st.session_state["chat_history"]:
        if msg["type"] == "human":
            # User messages - beautiful gradient bubbles
            st.markdown(
                f'<div class="user-bubble">{msg["content"]}</div>',
                unsafe_allow_html=True
            )
        elif msg["type"] == "ai":
            # AI responses - beautiful gradient bubbles
            st.markdown(
                f'<div class="ai-bubble">{msg["content"]}</div>',
                unsafe_allow_html=True
            )
    
    st.markdown('</div>', unsafe_allow_html=True)
else:
    # Welcome message for empty chat
    st.markdown("""
    <div style="text-align: center; padding: 50px; color: #666;">
        <h2>🌟 Welcome to IONOS RAG Chatbot!</h2>
        <p>Start a conversation by typing a message below.</p>
        <p>💡 <strong>Pro tip:</strong> Initialize a knowledge base first for better responses!</p>
    </div>
    """, unsafe_allow_html=True)

# --- Chat Input Form ---
st.markdown("---")

# Only show "Start a Conversation" title if no messages yet
if not st.session_state.get("chat_history", []):
    st.markdown("### 💭 Start a Conversation")

with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_message = st.text_input(
            "Type your message here...", 
            key="user_message",
            placeholder="Ask me anything! 🤔"
        )
    
    with col2:
        # Align button with input field
        st.markdown("<div style='margin-top: 10px;'>", unsafe_allow_html=True)
        send_btn = st.form_submit_button("🚀 Send", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# Handle message submission
if send_btn and user_message.strip():
    # Add user message to chat history immediately for better UX
    st.session_state["chat_history"].append({"type": "human", "content": user_message})
    
    # Send message to backend and get AI response
    with st.spinner("🤔 AI is thinking..."):
        try:
            resp = requests.post(
                f"{BACKEND_URL}/",
                json={"prompt": user_message},
                headers={"x-model-id": normalize_model_id(model)},
            )
            if resp.ok:
                # Parse the JSON response properly
                try:
                    response_data = resp.json()
                    ai_content = response_data.get("content", "No response content")
                except:
                    # Fallback to text if JSON parsing fails
                    ai_content = resp.text
                
                # Add AI response to chat history
                st.session_state["chat_history"].append({"type": "ai", "content": ai_content})
                
                # Attempt to sync with backend chat history
                try:
                    hist_resp = requests.get(f"{BACKEND_URL}/", headers={"x-model-id": normalize_model_id(model)})
                    if hist_resp.ok:
                        backend_history = hist_resp.json()
                        if len(backend_history) > len(st.session_state["chat_history"]):
                            st.session_state["chat_history"] = backend_history
                except Exception:
                    pass
                
                st.rerun()
            else:
                error_detail = resp.text
                if resp.status_code == 429:
                    st.markdown('<div class="error-message">⏰ Rate limit exceeded. Switch to a different model for immediate response!</div>', unsafe_allow_html=True)
                elif resp.status_code == 503:
                    st.markdown('<div class="error-message">🤖 Model temporarily unavailable. Switch to a different model for immediate response!</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="error-message">❌ Failed to get response: {error_detail}</div>', unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f'<div class="error-message">❌ Error: {str(e)}</div>', unsafe_allow_html=True)

