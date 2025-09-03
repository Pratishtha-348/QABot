# app.py
import streamlit as st
from chains.rag_chain import get_rag_chain_from_file

st.set_page_config(page_title="🤖 Q&A Bot", layout="centered")
st.title("🤖 Q&A Bot with Memory")

# Initialize session state
if "sessions" not in st.session_state:
    st.session_state.sessions = {"Session 1": []} 
if "current_session" not in st.session_state:
    st.session_state.current_session = "Session 1"
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Sidebar: manage sessions
with st.sidebar:
    st.header("🗂 Sessions")

    # Auto-generate next session name
    next_session_num = len(st.session_state.sessions) + 1
    new_session_name = f"Session {next_session_num}"


    # Button to create a new session automatically
    if st.button("➕ New Session"):
        session_count = len(st.session_state.sessions) + 1
        new_session_name = f"Session {session_count}"
        st.session_state.sessions[new_session_name] = []
        st.session_state.current_session = new_session_name
        st.session_state.chat_history = []
        st.session_state.rag_chain = None
        st.success(f"Created {new_session_name}")


    # Show session list
    session_choice = st.selectbox(
        "Switch session:",
        options=list(st.session_state.sessions.keys()),
        index=list(st.session_state.sessions.keys()).index(st.session_state.current_session),
    )

    if session_choice != st.session_state.current_session:
        st.session_state.current_session = session_choice
        st.session_state.chat_history = st.session_state.sessions.get(session_choice, [])
        st.session_state.rag_chain = None


# File uploader
uploaded_file = st.file_uploader("📂 Upload PDF/TXT/DOCX", type=["pdf", "txt", "docx", "doc"])

# Initialize chain when file uploaded
if uploaded_file and st.session_state.rag_chain is None:
    st.session_state.rag_chain = get_rag_chain_from_file(uploaded_file)


# Chat input

# Display chat history for the current session
for sender, msg in st.session_state.chat_history:
    if sender == "You":
        with st.chat_message("user"):
            st.markdown(msg)
    else:
        with st.chat_message("assistant"):
            st.markdown(msg)

# Chat input
user_input = st.chat_input("💬 Ask me something...")

if user_input and st.session_state.rag_chain:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.chat_history.append(("You", user_input))
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for chunk in st.session_state.rag_chain.stream({"question": user_input}):
            if isinstance(chunk, dict) and "answer" in chunk:
                token = chunk["answer"]
            else:
                token = str(chunk)
            full_response += token
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.chat_history.append(("Bot", full_response))
    st.session_state.sessions[st.session_state.current_session] = st.session_state.chat_history

