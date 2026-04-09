import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from streamlit_option_menu import option_menu
from api.llm import generate_response
from api.rag import process_document, create_vector_store, retrieve_context
from pypdf import PdfReader


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Enterprise AI Copilot", page_icon="🏦", layout="wide")

# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
<link href="https://cdn.jsdelivr.net/npm/bootstrap-icons/font/bootstrap-icons.css" rel="stylesheet">
<h1><i class="bi bi-bank"></i> Enterprise AI Copilot</h1>
""",
    unsafe_allow_html=True,
)
st.caption("Secure • Reliable • AI-powered knowledge assistant")


# -----------------------------
# Global State (IMPORTANT)
# -----------------------------
if "model" not in st.session_state:
    st.session_state.model = "llama3.2:3b"

if "temperature" not in st.session_state:
    st.session_state.temperature = 0.2

# -----------------------------
# RAG State
# -----------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# -----------------------------
# Sidebar (Enterprise Feel)
# -----------------------------
with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=["Chat", "Knowledge Base", "Settings"],
        icons=["chat", "database", "gear"],
        default_index=0,
    )

    st.divider()

    if selected == "Settings":
        st.subheader("Model Settings")

        available_models = ["llama3.2:3b", "llama3.2:7b", "llama3.2:1b"]

        new_model = st.selectbox(
            "Model",
            available_models,
            index=(
                available_models.index(st.session_state.model)
                if st.session_state.model in available_models
                else 0
            ),
        )

        new_temperature = st.slider(
            "Temperature", 0.0, 1.0, st.session_state.temperature
        )

        if new_model == "llama3.2:3b":
            st.info("🦙 **Llama 3.2 3B** - Good balance of speed and quality")
        elif new_model == "llama3.2:7b":
            st.info("🦙 **Llama 3.2 7B** - Higher quality, needs more RAM")
        elif new_model == "llama3.2:1b":
            st.info("🦙 **Llama 3.2 1B** - Very fast, lower quality")

        st.session_state.model = new_model
        st.session_state.temperature = new_temperature

    elif selected == "Knowledge Base":
        st.subheader("Documents")

        uploaded_file = st.file_uploader("Upload document", type=["pdf", "txt"])

        if uploaded_file:
            with st.spinner("Processing document..."):
                # Extract text
                if uploaded_file.type == "application/pdf":
                    reader = PdfReader(uploaded_file)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() or ""
                else:
                    text = uploaded_file.read().decode("utf-8")

                # Process + store
                chunks = process_document(text)
                st.session_state.vector_store = create_vector_store(chunks)

            st.success("Document processed successfully!")

    elif selected == "Chat":
        st.write("You're in chat mode")

# -----------------------------
# Chat State
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# Display Chat History
# -----------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------------------
# Chat Input
# -----------------------------
user_input = st.chat_input("Ask a question about policies, risk, or data...")

if user_input:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # If document exists → retrieve relevant chunks
    context = ""
    if st.session_state.vector_store:
        context = retrieve_context(user_input, st.session_state.vector_store)

    system_prompt = f"""
You are an enterprise AI assistant for banking and financial services.

Use the provided context to answer the question.
If the answer is not in the context, say you don't know.

Context:
{context}
"""

    messages_for_api = [
        {"role": "system", "content": system_prompt},
        *st.session_state.messages,
    ]

    # Call the LLM with current model and temperature settings
    with st.spinner("Generating response..."):
        response = generate_response(
            messages_for_api,
            model=st.session_state.model,
            temperature=st.session_state.temperature,
        )

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
