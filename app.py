import streamlit as st
import time
import requests
import uuid
from chains.rag_chain import get_rag_chain_from_file, get_rag_chain_from_url

# ----------------- CONFIG -----------------
API_URL = "http://localhost:8000/threadqa/chat"  # FastAPI chat endpoint

st.set_page_config(page_title="QABot", layout="wide")

# ----------------- SESSION STATE -----------------
if "sessions" not in st.session_state:
    st.session_state.sessions = {"Documents": {}, "URLs": {}}

if "current_tab" not in st.session_state:
    st.session_state.current_tab = "Documents"

if "current_chat" not in st.session_state:
    st.session_state.current_chat = None

if "history_fetched" not in st.session_state:
    st.session_state.history_fetched = {}

# ----------------- SIDEBAR: CHAT SESSIONS -----------------
with st.sidebar:
    st.header("💬 Sessions")
    active_sessions = st.session_state.sessions[st.session_state.current_tab]

    if st.button("➕ New Chat", use_container_width=True):
        new_name = f"Chat {len(active_sessions) + 1}"
        new_session_id = str(uuid.uuid4())
        active_sessions[new_name] = {
            "rag_chain": None,
            "messages": [],
            "session_id": new_session_id
        }
        st.session_state.current_chat = new_name
        st.session_state.history_fetched[new_name] = False

    for name in active_sessions.keys():
        if st.button(name, use_container_width=True):
            st.session_state.current_chat = name
            st.session_state.history_fetched[name] = False

# ----------------- TABS: DOCUMENT / URL -----------------
tab1, tab2 = st.tabs(["📄 Document Upload", "🔗 URL Upload"])

with tab1:
    st.session_state.current_tab = "Documents"
    st.subheader("Upload a document and chat")
    uploaded_file = st.file_uploader("Upload PDF, TXT, DOCX", type=["pdf", "txt", "docx"])

    if uploaded_file and st.button("Build RAG Chain from Document"):
        if st.session_state.current_chat is None:
            st.warning("⚠️ Please create a chat first!")
        else:
            with st.spinner("⏳ Processing document..."):
                time.sleep(2)
                active_sessions[st.session_state.current_chat]["rag_chain"] = get_rag_chain_from_file(
                    uploaded_file, st.session_state.current_chat
                )
            st.success("✅ Document RAG Chain is ready!")

with tab2:
    st.session_state.current_tab = "URLs"
    st.subheader("Enter a webpage URL and chat")
    url = st.text_input("Enter webpage URL")

    if url and st.button("Build RAG Chain from URL"):
        if st.session_state.current_chat is None:
            st.warning("⚠️ Please create a chat first!")
        else:
            with st.spinner("🌐 Fetching and processing URL..."):
                time.sleep(2)
                active_sessions[st.session_state.current_chat]["rag_chain"] = get_rag_chain_from_url(
                    url, st.session_state.current_chat
                )
            st.success("✅ URL RAG Chain is ready!")

# ----------------- CHAT AREA -----------------
active_sessions = st.session_state.sessions[st.session_state.current_tab]
if st.session_state.current_chat and st.session_state.current_chat in active_sessions:
    st.subheader("💬 Chat")
    chat = active_sessions[st.session_state.current_chat]
    session_id = chat["session_id"]

    # 1️⃣ Fetch chat history (only once per chat)
    if not st.session_state.history_fetched.get(st.session_state.current_chat, False):
        resp = requests.get(f"{API_URL}/{session_id}")
        if resp.status_code == 200:
            chat["messages"] = resp.json()
        st.session_state.history_fetched[st.session_state.current_chat] = True

    # 2️⃣ Display messages
    for msg in chat["messages"]:
        if msg["question"]:
            with st.chat_message("user"):
                st.markdown(msg["question"])
        if msg["answer"]:
            with st.chat_message("assistant"):
                st.markdown(msg["answer"])

    # 3️⃣ Chat input
    if query := st.chat_input("Ask your question..."):
        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(query)

        # 4️⃣ Generate answer first
        if chat.get("rag_chain"):
            with st.spinner("🤖 Thinking..."):
                response = chat["rag_chain"].invoke({"question": query})
                if "chat_history" in response:
                    last_msg = response["chat_history"][-1]
                    answer = getattr(last_msg, "content", str(last_msg))
                else:
                    answer = response.get("answer", "⚠️ No answer generated")

            # 5️⃣ Show answer with typing animation
            with st.chat_message("assistant"):
                placeholder = st.empty()
                typed_text = ""
                for char in answer:
                    typed_text += char
                    placeholder.markdown(typed_text)
                    time.sleep(0.02)

            # 6️⃣ Insert full QA into DB in ONE shot
            payload = {
                "session_id": session_id,
                "label": st.session_state.current_chat,
                "question": query,
                "answer": answer
            }
            save_resp = requests.post(API_URL, json=payload)

            if save_resp.status_code == 200:
                chat["messages"].append(payload)
        else:
            st.warning("⚠️ Please build a RAG chain first.")
