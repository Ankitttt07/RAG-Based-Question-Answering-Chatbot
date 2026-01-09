import os

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_DIR = "data"

if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(
        "Create a 'data' folder and add PDF, TXT, or DOCX files."
    )

documents = []

for file in os.listdir(DATA_DIR):
    file_path = os.path.join(DATA_DIR, file)

    if file.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file.endswith(".txt"):
        loader = TextLoader(file_path)
    elif file.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        continue

    documents.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")

print("Step 1 completed successfully. FAISS index created.")
