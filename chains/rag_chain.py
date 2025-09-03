import os
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from chains.vectorstore_loader import load_vectorstore


def get_rag_chain_from_file(uploaded_file, session_name="default"):
    """
    Build a session-specific RAG chain with memory + Chroma retriever.
    Each session will have its own Chroma DB and chat history.
    """

    # Create session-specific folder for Chroma persistence
    persist_dir = os.path.join("chroma_dbs", session_name)
    os.makedirs(persist_dir, exist_ok=True)

    # Load vectorstore (Chroma + embeddings) for this session
    vectorstore = load_vectorstore(uploaded_file, persist_directory=persist_dir)

    # Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Memory (for conversation context)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    # Gemini LLM
    from utils.config import GOOGLE_API_KEY
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        google_api_key=GOOGLE_API_KEY  
    )

    # Conversational RAG chain
    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer"
    )

    return rag_chain
