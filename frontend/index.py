import streamlit as st
from streamlit_option_menu import option_menu


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Enterprise AI Copilot",
    page_icon="🏦",
    layout="wide"
)

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<link href="https://cdn.jsdelivr.net/npm/bootstrap-icons/font/bootstrap-icons.css" rel="stylesheet">
<h1><i class="bi bi-bank"></i> Enterprise AI Copilot</h1>
""", unsafe_allow_html=True)
st.caption("Secure • Reliable • AI-powered knowledge assistant")

# -----------------------------
# Sidebar (Enterprise Feel)
# -----------------------------
with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=["Chat", "Knowledge Base", "Settings"],
        icons=["chat", "database", "gear"],  # Bootstrap icons
        default_index=0,
    )

    st.divider()

    if selected == "Settings":
        st.subheader("Model Settings")

        model = st.selectbox(
            "Model",
            ["gpt-4.1-mini", "gpt-4.1", "mock-model"]
        )

        temperature = st.slider(
            "Temperature",
            0.0, 1.0, 0.2
        )

    elif selected == "Knowledge Base":
        st.subheader("Documents")

        uploaded_file = st.file_uploader(
            "Upload document",
            type=["pdf", "txt"]
        )

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
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # Placeholder response (Phase 2 will replace this)
    response = f"""
👋 This is a placeholder response.

You asked:
> {user_input}

⚙️ Model: {model}  
🌡️ Temperature: {temperature}

➡️ Next step: connect to LLM backend.
"""

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })