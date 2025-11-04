"""
chains/vectorstore_loader.py
----------------------------
Load documents and create vector stores with support for both file paths and Streamlit UploadedFile.
"""

import os
import tempfile
from typing import Optional, Union
from langchain_community.document_loaders import PyPDFLoader, UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ============================================================================
# Helper: Handle Both File Paths and Streamlit UploadedFile Objects
# ============================================================================

def _get_file_path(file_input) -> str:
    """
    Convert input to a file path string.
    Handles both regular file paths (str) and Streamlit UploadedFile objects.
    
    Args:
        file_input: Either a string file path or Streamlit UploadedFile object
        
    Returns:
        str: File path that can be used with loaders
    """
    # If it's already a string path, return it as-is
    if isinstance(file_input, str):
        return file_input
    
    # If it's a Streamlit UploadedFile, save it to a temp file
    if hasattr(file_input, 'read') and hasattr(file_input, 'name'):
        # Get the file extension from the uploaded file name
        file_extension = os.path.splitext(file_input.name)[1]
        
        # Create a temporary file with the same extension
        temp_file = tempfile.NamedTemporaryFile(
            delete=False, 
            suffix=file_extension
        )
        
        # Write the uploaded file content to the temp file
        temp_file.write(file_input.read())
        temp_file.close()
        
        return temp_file.name
    
    # If it's neither, raise an error
    raise ValueError(f"Unsupported input type: {type(file_input)}")

# ============================================================================
# Main Loading Functions (Minimal Changes)
# ============================================================================

def load_vectorstore(file_input: Union[str, object], persist_directory: Optional[str] = None):
    """
    Load documents from a file and create a vector store.
    
    Args:
        file_input: Either a file path (str) or Streamlit UploadedFile object
        persist_directory: Directory to persist the vector store
    
    Returns:
        Chroma: A vector store containing the document embeddings
    """
    # ✅ ADDED: Convert to file path (handles both str and UploadedFile)
    file_path = _get_file_path(file_input)
    
    # Load the document (rest of your code unchanged)
    if file_path.lower().endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    else:
        raise ValueError("Unsupported file type. Currently only PDF files are supported.")
    
    documents = loader.load()
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    texts = text_splitter.split_documents(documents)
    
    # Initialize embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    return vectorstore


def load_vectorstore_from_url(url: str, persist_directory: Optional[str] = None):
    """
    Load documents from a URL and create a vector store.
    
    Args:
        url: URL to fetch content from
        persist_directory: Directory to persist the vector store
    
    Returns:
        Chroma: A vector store containing the document embeddings
    """
    # Load the document from URL
    loader = UnstructuredURLLoader([url])
    documents = loader.load()
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    texts = text_splitter.split_documents(documents)
    
    # Initialize embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    return vectorstore