import os
from PyPDF2 import PdfReader
import docx
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def load_vectorstore(uploaded_file, persist_directory="chroma_dbs/default"):
    ext = os.path.splitext(uploaded_file.name)[1].lower()

    # PDF -> read directly from file-like object
    if ext == ".pdf":
        reader = PdfReader(uploaded_file)
        docs = [Document(page_content=page.extract_text()) for page in reader.pages]

    # TXT -> decode directly
    elif ext == ".txt":
        content = uploaded_file.read().decode("utf-8")
        docs = [Document(page_content=content)]

    # DOCX/DOC -> use python-docx
    elif ext in [".docx", ".doc"]:
        doc = docx.Document(uploaded_file)
        content = "\n".join([p.text for p in doc.paragraphs])
        docs = [Document(page_content=content)]

    else:
        raise ValueError("Unsupported file format")

    # HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Persist Chroma DB
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    return vectorstore
