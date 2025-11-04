"""
app.py
------
Streamlit UI for Advanced RAG Agent with RL Feedback
"""

import streamlit as st
import time
import requests
import uuid
from chains.rag_agent import get_agentic_rag_from_file, get_agentic_rag_from_url, agent_with_feedback

# ----------------- CONFIG -----------------
API_URL = "http://localhost:8000/threadqa/chat"  # FastAPI chat endpoint

st.set_page_config(page_title="🤖 Advanced QABot", layout="wide")

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
def regenerate_answer_with_typing_effect(agent, question, placeholder, feedback_placeholder=None):
    """Generate answer with typing effect and RL feedback"""
    try:
        # Use the agent_with_feedback function for RL evaluation
        answer, feedback = agent_with_feedback(agent, question)
        
        # Typing effect for answer
        typed_text = ""
        for char in answer:
            typed_text += char
            placeholder.markdown(typed_text)
            time.sleep(0.02)
        
        # Display feedback metrics if placeholder provided
        if feedback_placeholder:
            display_feedback_metrics(feedback, feedback_placeholder)
        
        return answer, feedback
    except Exception as e:
        error_msg = f"❌ Error generating answer: {str(e)}"
        placeholder.markdown(error_msg)
        return error_msg, None

def display_feedback_metrics(feedback, container):
    """Display RL feedback metrics in a nice format"""
    if not feedback:
        return
    
    with container:
        score = feedback["score"]
        score_color = "🟢" if score >= 0.8 else "🟡" if score >= 0.7 else "🔴"
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Overall", f"{score_color} {score:.2%}")
        with col2:
            st.metric("Length", f"{feedback['metrics']['length_score']:.0%}")
        with col3:
            st.metric("Relevance", f"{feedback['metrics']['relevance_score']:.0%}")
        with col4:
            st.metric("Confidence", f"{feedback['metrics']['confidence_score']:.0%}")
        with col5:
            st.metric("Structure", f"{feedback['metrics']['structure_score']:.0%}")
        
        if feedback['num_attempts'] > 1:
            st.caption(f"🔄 Refined {feedback['num_attempts']} times for quality")

def save_qa_to_database(session_id, label, question, answer, feedback=None, original_msg_id=None):
    """Save Q&A to database with optional feedback scores"""
    payload = {
        "session_id": session_id,
        "label": label,
        "question": question,
        "answer": answer,
        "original_msg_id": original_msg_id,
        "feedback": feedback  # Include RL feedback
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

# ----------------- HEADER -----------------
st.title("🤖 RAG Agent")
st.markdown("Upload documents or URLs and chat with an intelligent agent that self-evaluates its answers.")

# ----------------- SIDEBAR: CHAT SESSIONS -----------------
with st.sidebar:
    st.header("💬 Sessions")
    active_sessions = st.session_state.sessions[st.session_state.current_tab]

    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        new_name = f"Chat {len(active_sessions) + 1}"
        new_session_id = str(uuid.uuid4())
        active_sessions[new_name] = {
            "agent": None,  # Changed from rag_chain to agent
            "messages": [],
            "session_id": new_session_id
        }
        st.session_state.current_chat = new_name
        st.session_state.history_fetched[new_name] = False
        st.rerun()

    st.divider()
    
    # Display existing sessions
    for name in active_sessions.keys():
        is_current = name == st.session_state.current_chat
        if st.button(
            name, 
            use_container_width=True,
            type="primary" if is_current else "secondary"
        ):
            st.session_state.current_chat = name
            st.session_state.history_fetched[name] = False
            st.rerun()

# ----------------- TABS: DOCUMENT / URL -----------------
tab1, tab2 = st.tabs(["📄 Document Upload", "🔗 URL Upload"])

with tab1:
    st.session_state.current_tab = "Documents"
    st.subheader("Upload a document and chat")
    uploaded_file = st.file_uploader("Upload PDF, TXT, DOCX", type=["pdf"])  # Currently only PDF supported
    
    if uploaded_file:
        if st.button("🚀 Build Agent from Document", type="primary"):
            if st.session_state.current_chat is None:
                st.warning("⚠️ Please create a chat first!")
            else:
                with st.spinner("⏳ Processing document and building agent..."):
                    try:
                        # Use the agent version instead of plain RAG chain
                        agent = get_agentic_rag_from_file(
                            uploaded_file, 
                            session_name=st.session_state.current_chat
                        )
                        active_sessions[st.session_state.current_chat]["agent"] = agent
                        st.success("✅ Advanced Agent is ready! The agent has 4 specialized tools and RL feedback.")
                        st.balloons()
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

with tab2:
    st.session_state.current_tab = "URLs"
    st.subheader("Enter a webpage URL and chat")
    url = st.text_input("Enter webpage URL", placeholder="https://example.com")
    
    if url:
        if st.button("🚀 Build Agent from URL", type="primary"):
            if st.session_state.current_chat is None:
                st.warning("⚠️ Please create a chat first!")
            else:
                with st.spinner("🌐 Fetching and processing URL..."):
                    try:
                        # Use the agent version
                        agent = get_agentic_rag_from_url(
                            url, 
                            session_name=st.session_state.current_chat
                        )
                        active_sessions[st.session_state.current_chat]["agent"] = agent
                        st.success("✅ Advanced Agent is ready! The agent has 4 specialized tools and RL feedback.")
                        st.balloons()
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

# ----------------- CHAT AREA -----------------
active_sessions = st.session_state.sessions[st.session_state.current_tab]

if st.session_state.current_chat and st.session_state.current_chat in active_sessions:
    chat = active_sessions[st.session_state.current_chat]
    session_id = chat["session_id"]
    
    # Show agent status
    if chat.get("agent"):
        st.success("🟢 Agent Active | 🧠 4 Tools Available | 📊 RL Feedback Enabled")
    else:
        st.warning("🔴 No Agent - Please upload a document or URL above")
    
    st.subheader("💬 Chat")

    # 1️⃣ Fetch chat history (only once per chat)
    if not st.session_state.history_fetched.get(st.session_state.current_chat, False):
        try:
            resp = requests.get(f"{API_URL}/{session_id}")
            if resp.status_code == 200:
                chat["messages"] = resp.json()
        except Exception as e:
            # It's okay if the API is not running
            pass
        st.session_state.history_fetched[st.session_state.current_chat] = True

    # 2️⃣ Display messages
    for msg in chat["messages"]:
        msg_id = msg.get("id")
        is_edited = msg.get("is_edited", False)
        
        # Display question
        if msg["question"]:
            with st.chat_message("user"):
                if is_edited:
                    st.markdown(f"✏️ **(Edited)** {msg['question']}")
                else:
                    st.markdown(msg["question"])
                
                # Edit button
                if st.session_state.editing_msg_id != msg_id:
                    if st.button("✏️ Edit", key=f"edit_question_{msg_id}"):
                        st.session_state.editing_msg_id = msg_id
                        st.session_state.editing_text = msg["question"]
                        st.rerun()

        # Display answer with feedback
        if msg.get("answer"):
            with st.chat_message("assistant"):
                st.markdown(msg["answer"])
                
                # Show feedback metrics if available
                if msg.get("feedback"):
                    with st.expander("📊 Answer Quality Metrics", expanded=False):
                        display_feedback_metrics(msg["feedback"], st.container())

    # 3️⃣ Edit Section with Answer Regeneration
    if st.session_state.editing_msg_id is not None:
        st.divider()
        st.subheader("✏️ Edit Question")
        
        # Find the original message
        original_msg = None
        for m in chat["messages"]:
            if m.get("id") == st.session_state.editing_msg_id:
                original_msg = m
                break
        
        if original_msg:
            col1, col2 = st.columns([4, 1])
            
            with col1:
                new_question = st.text_input(
                    "Edit your question:",
                    value=st.session_state.editing_text,
                    key="edit_question_input"
                )
            
            with col2:
                st.write("")  # Spacing
                if st.button("❌ Cancel", key="cancel_edit"):
                    st.session_state.editing_msg_id = None
                    st.session_state.editing_text = ""
                    st.rerun()
            
            # Regenerate button
            if st.button(
                "🔄 Regenerate Answer with Agent", 
                key="regenerate_answer",
                disabled=st.session_state.regenerating_answer or not new_question.strip(),
                type="primary"
            ):
                if not chat.get("agent"):
                    st.error("❌ No agent available! Please build an agent first.")
                else:
                    st.session_state.regenerating_answer = True
                    
                    # Generate new answer
                    with st.spinner("🤖 Agent is thinking and evaluating..."):
                        # Show the edited question
                        with st.chat_message("user"):
                            st.markdown(f"✏️ **(Edited)** {new_question.strip()}")
                        
                        # Generate answer with typing effect
                        with st.chat_message("assistant"):
                            answer_placeholder = st.empty()
                            feedback_placeholder = st.container()
                            
                            answer, feedback = regenerate_answer_with_typing_effect(
                                chat["agent"], 
                                new_question.strip(),
                                answer_placeholder,
                                feedback_placeholder
                            )
                    
                    # Save to database
                    if answer and not answer.startswith("❌"):
                        saved_msg = save_qa_to_database(
                            session_id,
                            st.session_state.current_chat,
                            new_question.strip(),
                            answer,
                            feedback,
                            original_msg_id=original_msg.get("id")
                        )
                        
                        if saved_msg:
                            saved_msg["is_edited"] = True
                            chat["messages"].append(saved_msg)
                            st.success("✅ Answer regenerated with quality evaluation!")
                            time.sleep(1)
                    
                    # Reset editing state
                    st.session_state.regenerating_answer = False
                    st.session_state.editing_msg_id = None
                    st.session_state.editing_text = ""
                    st.rerun()

    # 4️⃣ Chat input
    if st.session_state.editing_msg_id is None:
        if query := st.chat_input("Ask your question..."):
            # Display user question
            with st.chat_message("user"):
                st.markdown(query)

            if chat.get("agent"):
                # Generate answer with agent
                with st.chat_message("assistant"):
                    answer_placeholder = st.empty()
                    answer_placeholder.markdown("🤖 Agent is thinking...")
                    
                    # Container for feedback metrics
                    feedback_container = st.container()
                    
                    # Get answer with RL feedback
                    answer, feedback = agent_with_feedback(chat["agent"], query)
                    
                    # Display answer with typing effect
                    typed_text = ""
                    for char in answer:
                        typed_text += char
                        answer_placeholder.markdown(typed_text)
                        time.sleep(0.01)
                    
                    # Display feedback metrics
                    with st.expander("📊 Answer Quality Metrics", expanded=True):
                        display_feedback_metrics(feedback, st.container())

                # Save to database
                saved_msg = save_qa_to_database(
                    session_id, 
                    st.session_state.current_chat, 
                    query, 
                    answer,
                    feedback
                )

                if saved_msg:
                    chat["messages"].append(saved_msg)
            else:
                st.warning("⚠️ Please build an agent first by uploading a document or URL.")
else:
    # Welcome screen
    st.info("👈 Create or select a chat session to get started!")
    
    st.markdown("""
    ### 🌟 Features of this Advanced RAG System
    
    - **🧠 Multi-Tool Agent**: Intelligently uses 4 specialized tools:
        - RAGRetriever, Summarizer, Explainer, Comparator
    - **📊 RL Feedback**: Every answer is automatically evaluated on:
        - Length, Relevance, Confidence, Structure
    - **🔄 Auto-Refinement**: Answers below 70% quality are automatically improved
    - **✏️ Edit & Regenerate**: Edit your questions and get new answers
    - **💾 Persistent History**: All conversations are saved
    """)

# Footer
st.divider()
st.caption("🧠 Powered by Google Gemini 1.5 + LangChain Agent + RL Feedback System")