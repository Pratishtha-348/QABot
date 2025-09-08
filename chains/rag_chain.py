import os
import warnings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from chains.vectorstore_loader import load_vectorstore, load_vectorstore_from_url
from utils.config import GOOGLE_API_KEY

# 🔹 Suppress noisy LangChain deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 🔹 Build RAG chain
def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        google_api_key=GOOGLE_API_KEY
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer"
    )

# 🔹 File input
def get_rag_chain_from_file(uploaded_file, session_name="default"):
    if not session_name:
        session_name = "default_file"

    persist_dir = os.path.join("chroma_dbs", f"{session_name}_file")
    os.makedirs(persist_dir, exist_ok=True)

    vectorstore = load_vectorstore(uploaded_file, persist_directory=persist_dir)
    return build_rag_chain(vectorstore)

# 🔹 URL input
def get_rag_chain_from_url(url, session_name="default"):
    if not session_name:
        session_name = "default_url"

    persist_dir = os.path.join("chroma_dbs", f"{session_name}_url")
    os.makedirs(persist_dir, exist_ok=True)

    vectorstore = load_vectorstore_from_url(url, persist_directory=persist_dir)
    return build_rag_chain(vectorstore)
