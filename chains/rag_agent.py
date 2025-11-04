"""
chains/rag_agent.py
-------------------
End-to-end "agent-augmented RAG" utility module with RL feedback.
"""

from __future__ import annotations
import os
import warnings
from typing import Dict, Tuple

# LangChain imports
try:
    from langchain.memory import ConversationBufferMemory
except ModuleNotFoundError:
    from langchain_community.memory import ConversationBufferMemory

try:
    from langchain.chains import ConversationalRetrievalChain
except ModuleNotFoundError:
    from langchain_community.chains import ConversationalRetrievalChain

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, Tool, AgentType

from chains.vectorstore_loader import load_vectorstore, load_vectorstore_from_url
from utils.config import GOOGLE_API_KEY

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ============================================================================
# 1) PLAIN RAG CHAIN (Foundation)
# ============================================================================

def _build_rag_chain(vectorstore):
    """Conversational RAG with Gemini 1.5-Flash."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 7})  # ✅ Tuned for better retrieval
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        google_api_key=GOOGLE_API_KEY,
        convert_system_message_to_human=True,  # ✅ Critical fix
    )
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer",
    )

def get_rag_chain_from_file(uploaded_file: str, session_name: str = "default"):
    persist_dir = os.path.join("chroma_dbs", f"{session_name or 'default'}_file")
    os.makedirs(persist_dir, exist_ok=True)
    vectorstore = load_vectorstore(uploaded_file, persist_directory=persist_dir)
    return _build_rag_chain(vectorstore)

def get_rag_chain_from_url(url: str, session_name: str = "default"):
    persist_dir = os.path.join("chroma_dbs", f"{session_name or 'default'}_url")
    os.makedirs(persist_dir, exist_ok=True)
    vectorstore = load_vectorstore_from_url(url, persist_directory=persist_dir)
    return _build_rag_chain(vectorstore)

# ============================================================================
# 2) AGENT WITH MULTIPLE SPECIALIZED TOOLS
# ============================================================================

def create_agent_from_rag_chain(rag_chain):
    """Wrap RAG with intelligent multi-tool agent."""
    
    # Define 4 specialized tools
    rag_tool = Tool(
        name="RAGRetriever",
        func=lambda q: rag_chain.invoke({"question": q})["answer"],
        description="Fact-based questions. Always use bullet points in answers.",
    )
    
    summarize_tool = Tool(
        name="Summarizer",
        func=lambda q: rag_chain.invoke(
            {"question": f"Provide a concise summary of: {q}"})["answer"],
        description="Condense long content into key points.",
    )
    
    explain_tool = Tool(
        name="Explainer",
        func=lambda q: rag_chain.invoke(
            {"question": f"Explain with examples: {q}"})["answer"],
        description="Detailed explanations with examples.",
    )
    
    compare_tool = Tool(
        name="Comparator",
        func=lambda q: rag_chain.invoke(
            {"question": f"Compare and contrast: {q}"})["answer"],
        description="Highlight similarities and differences.",
    )
    
    # Agent controller LLM (more powerful model)
    controller_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=GOOGLE_API_KEY,
        convert_system_message_to_human=True,
    )
    
    # Agent memory
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True
    )
    
    # Safe agent type selection
    safe_agent_type = (
        AgentType.__members__.get("CHAT_CONVERSATIONAL_REACT_DESCRIPTION")
        or AgentType.ZERO_SHOT_REACT_DESCRIPTION
    )
    
    # Initialize agent
    agent = initialize_agent(
        tools=[rag_tool, summarize_tool, explain_tool, compare_tool],
        llm=controller_llm,
        agent=safe_agent_type,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3,
        early_stopping_method="generate",
    )
    return agent

# ============================================================================
# 3) RL FEEDBACK LOOP (Automatic Quality Evaluation & Retry)
# ============================================================================

def feedback_evaluator(answer: str, query: str) -> Tuple[float, Dict]:
    """Evaluate answer quality on multiple dimensions."""
    details = {
        "length_score": 0.0,
        "relevance_score": 0.0,
        "confidence_score": 0.0,
        "structure_score": 0.0,
    }
    
    # Length scoring
    n_chars = len(answer.strip())
    if n_chars >= 150: details["length_score"] = 1.0
    elif n_chars >= 80: details["length_score"] = 0.6
    else: details["length_score"] = 0.2
    
    # Relevance (keyword overlap)
    q_tokens = set(query.lower().split())
    a_tokens = set(answer.lower().split())
    overlap = len(q_tokens & a_tokens)
    details["relevance_score"] = min(1.0, overlap / max(1, len(q_tokens)))
    
    # Confidence (penalize hedging)
    uncertainty_phrases = [
        "not sure", "i am unsure", "i don't know", "uncertain",
        "maybe", "probably", "possibly", "guess"
    ]
    penalty = sum(0.1 for p in uncertainty_phrases if p in answer.lower())
    details["confidence_score"] = max(0.0, 1.0 - penalty)
    
    # Structure (prefer formatted answers)
    details["structure_score"] = (
        1.0 if any(sym in answer for sym in ["•", "-", "1.", "2."]) else 0.4
    )
    
    overall = sum(details.values()) / len(details)
    return overall, details

def agent_with_feedback(
    agent, query: str, min_score: float = 0.7, max_retries: int = 2
):
    """Execute agent with automatic quality feedback and retry."""
    attempts = 0
    
    while attempts <= max_retries:
        attempts += 1
        answer = agent.run(query)
        score, metrics = feedback_evaluator(answer, query)
        
        # Return if quality threshold met or max retries reached
        if score >= min_score or attempts > max_retries:
            return answer, {
                "score": score,
                "metrics": metrics,
                "num_attempts": attempts,
            }
        
        # Improvement prompt for retry
        query = (
            f"The previous answer scored {score:.2f}/1.0. Metrics: {metrics}\n"
            "Please refine: make it longer (≥150 chars), more relevant, "
            "confident, and well-structured (use bullets)."
        )
    
    return answer, {"score": score, "metrics": metrics, "num_attempts": attempts}

# ============================================================================
# 4) CONVENIENCE FUNCTIONS (Entry Points)
# ============================================================================

def get_agentic_rag_from_file(uploaded_file: str, session_name: str = "default"):
    """Build full agent system from file."""
    rag_chain = get_rag_chain_from_file(uploaded_file, session_name)
    return create_agent_from_rag_chain(rag_chain)

def get_agentic_rag_from_url(url: str, session_name: str = "default"):
    """Build full agent system from URL."""
    rag_chain = get_rag_chain_from_url(url, session_name)
    return create_agent_from_rag_chain(rag_chain)