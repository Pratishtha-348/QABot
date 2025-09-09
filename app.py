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

# Track editing state
if "editing_msg_id" not in st.session_state:
    st.session_state.editing_msg_id = None
if "editing_text" not in st.session_state:
    st.session_state.editing_text = ""
if "regenerating_answer" not in st.session_state:
    st.session_state.regenerating_answer = False

# ----------------- HELPER FUNCTIONS -----------------
def regenerate_answer_with_typing_effect(rag_chain, question, placeholder):
    """Generate answer with typing effect"""
    try:
        response = rag_chain.invoke({"question": question})
        
        if "chat_history" in response:
            last_msg = response["chat_history"][-1]
            answer = getattr(last_msg, "content", str(last_msg))
        else:
            answer = response.get("answer", "⚠️ No answer generated")
        
        # Typing effect
        typed_text = ""
        for char in answer:
            typed_text += char
            placeholder.markdown(typed_text)
            time.sleep(0.02)
        
        return answer
    except Exception as e:
        error_msg = f"❌ Error generating answer: {str(e)}"
        placeholder.markdown(error_msg)
        return error_msg

def save_qa_to_database(session_id, label, question, answer, original_msg_id=None):
    """Save Q&A to database and return the saved message"""
    payload = {
        "session_id": session_id,
        "label": label,
        "question": question,
        "answer": answer,
        "original_msg_id": original_msg_id  # For tracking edits
    }
    
    try:
        save_resp = requests.post(API_URL, json=payload)
        if save_resp.status_code == 200:
            return save_resp.json()
        else:
            st.error(f"Failed to save to database: {save_resp.status_code}")
            return None
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return None

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
        try:
            resp = requests.get(f"{API_URL}/{session_id}")
            if resp.status_code == 200:
                chat["messages"] = resp.json()
        except Exception as e:
            st.error(f"Failed to fetch chat history: {str(e)}")
        st.session_state.history_fetched[st.session_state.current_chat] = True

    # 2️⃣ Display messages
    for msg in chat["messages"]:
        msg_id = msg.get("id")
        is_edited = msg.get("is_edited", False)
        
        if msg["question"]:
            with st.chat_message("user"):
                # Show edited indicator
                if is_edited:
                    st.markdown(f"✏️ **(Edited)** {msg['question']}")
                else:
                    st.markdown(msg["question"])
                
                # Show Edit button for question (only if not currently editing this message)
                if st.session_state.editing_msg_id != msg_id:
                    if st.button("✏️ Edit", key=f"edit_question_{msg_id}"):
                        st.session_state.editing_msg_id = msg_id
                        st.session_state.editing_text = msg["question"]
                        st.rerun()

        if msg.get("answer"):
            with st.chat_message("assistant"):
                st.markdown(msg["answer"])

    # 3️⃣ Enhanced Edit Section with Answer Regeneration
    if st.session_state.editing_msg_id is not None:
        st.divider()
        st.subheader("✏️ Edit Question")
        
        # Find the original message
        original_msg = None
        for m in chat["messages"]:
            if m["id"] == st.session_state.editing_msg_id:
                original_msg = m
                break
        
        if original_msg:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                new_question = st.text_input(
                    "Edit your question:",
                    value=st.session_state.editing_text,
                    key="edit_question_input"
                )
            
            with col2:
                st.write("")  # Spacing
                cancel_edit = st.button("❌ Cancel", key="cancel_edit")
            
            # Action buttons
            col1, col2 = st.columns(2)
            
            with col1:
                regenerate_btn = st.button(
                    "🔄 Regenerate Answer", 
                    key="regenerate_answer",
                    disabled=st.session_state.regenerating_answer or not new_question.strip(),
                    type="primary"
                )
            
           
            # Handle cancel
            if cancel_edit:
                st.session_state.editing_msg_id = None
                st.session_state.editing_text = ""
                st.rerun()
            
           
          
            # Handle regenerate answer
            if regenerate_btn and new_question.strip():
                if not chat.get("rag_chain"):
                    st.error("❌ No RAG chain available! Please build a RAG chain first.")
                else:
                    st.session_state.regenerating_answer = True
                    print(st.session_state,'------------------st.session_state.')
                    latest_msg_id = None
                    if chat["messages"]:
                        latest_msg_id = chat["messages"][-1]["id"]

                    with st.spinner("🤖 Regenerating answer..."):
                        # Show the new question immediately
                        with st.chat_message("user"):
                            st.markdown(f"✏️ **(Edited)** {new_question.strip()}")
                        
                        # Generate new answer with typing effect
                        with st.chat_message("assistant"):
                            answer_placeholder = st.empty()
                            answer_placeholder.markdown("🤖 Thinking...")
                            
                            new_answer = regenerate_answer_with_typing_effect(
                                chat["rag_chain"], 
                                new_question.strip(),
                                answer_placeholder
                            )
                    
                    # Save to database as new entry
                    if not new_answer.startswith("❌"):
                        saved_msg = save_qa_to_database(
                            session_id,
                            st.session_state.current_chat,
                            new_question.strip(),
                            new_answer,
                            original_msg_id=latest_msg_id
                        )
                        
                        print(saved_msg,'------------------saved_msg.')
                        if saved_msg:
                            saved_msg["is_edited"] = True
                            chat["messages"].append(saved_msg)
                            st.success("✅ Question and answer updated successfully!")
                            time.sleep(2)
                        else:
                            st.error("❌ Failed to save to database.")
                        
                        print(saved_msg,'------------------saved_msg.')
                    else:
                        st.error("❌ Failed to generate new answer.")
                    
                    # Reset editing state
                    st.session_state.regenerating_answer = False
                    st.session_state.editing_msg_id = None
                    st.session_state.editing_text = ""
                    st.rerun()

    # 4️⃣ Chat input (only show if not editing)
    if st.session_state.editing_msg_id is None:
        if query := st.chat_input("Ask your question..."):
            with st.chat_message("user"):
                st.markdown(query)

            if chat.get("rag_chain"):
                with st.spinner("🤖 Thinking..."):
                    response = chat["rag_chain"].invoke({"question": query})
                    if "chat_history" in response:
                        last_msg = response["chat_history"][-1]
                        answer = getattr(last_msg, "content", str(last_msg))
                    else:
                        answer = response.get("answer", "⚠️ No answer generated")

                # Show answer with typing effect
                with st.chat_message("assistant"):
                    placeholder = st.empty()
                    typed_text = ""
                    for char in answer:
                        typed_text += char
                        placeholder.markdown(typed_text)
                        time.sleep(0.02)

                # Save to DB
                saved_msg = save_qa_to_database(session_id, st.session_state.current_chat, query, answer)

                if saved_msg:
                    chat["messages"].append(saved_msg)
            else:
                st.warning("⚠️ Please build a RAG chain first.")
else:
    st.info("👈 Create or select a chat session to get started!")