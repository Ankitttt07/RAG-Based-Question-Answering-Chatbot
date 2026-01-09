import os
from dotenv import load_dotenv

from google import genai

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found")

api_key = api_key.strip()  # REMOVES tabs, spaces, newlines

client = genai.Client(api_key=api_key)

# Load embeddings (must match Step-1)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS index
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


def ask_question(question: str) -> str:
    # NEW API (important)
    docs = retriever.invoke(question)

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
You are a document-based assistant.
Answer ONLY using the context below.
If the answer is not present, say "Answer not available in the documents."

Context:
{context}

Question:
{question}
"""

    response = client.models.generate_content(
        model="	gemini-2.0-flash-lite   ",
        contents=prompt
    )

    return response.text
