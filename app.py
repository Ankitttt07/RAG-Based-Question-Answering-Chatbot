import streamlit as st
from rag_engine import ask_question

st.set_page_config(page_title="RAG QA Chatbot")

st.title("📄 Multi-Document RAG Chatbot")
st.write("Supports PDF, TXT, and Word files")

question = st.text_input("Enter your question")

if st.button("Get Answer"):
    if question.strip():
        with st.spinner("Searching documents..."):
            answer = ask_question(question)
            st.success(answer)
    else:
        st.warning("Please enter a question")
