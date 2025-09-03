import os
import tempfile
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_huggingface import HuggingFaceEmbeddings


def load_vectorstore(uploaded_file, persist_directory="chroma_dbs/default"):
    """
    Load (or create) a Chroma vectorstore from uploaded file with persistence.
    If DB already exists, load it instead of rebuilding.
    """

    # ✅ Save UploadedFile to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    # Pick loader based on file type
    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(tmp_file_path)
    elif uploaded_file.name.endswith(".txt"):
        loader = TextLoader(tmp_file_path)
    elif uploaded_file.name.endswith(".docx") or uploaded_file.name.endswith(".doc"):
        loader = Docx2txtLoader(tmp_file_path)
    else:
        raise ValueError("Unsupported file format")

    docs = loader.load()

    # ✅ HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Persist Chroma DB
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    # Cleanup temp file (optional, but safe to keep until persist finishes)
    os.remove(tmp_file_path)

    return vectorstore
