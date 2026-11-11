import io
import os

import pdfplumber
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()  # reads GOOGLE_API_KEY from a local .env file (never commit it)

EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
CHAT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant answering questions about a PDF document.\n\n"
     "Guidelines:\n"
     "1. Provide complete, well-explained answers using the context below.\n"
     "2. Include relevant details, numbers, and explanations to give a thorough response.\n"
     "3. If the context mentions related information, include it to give a fuller picture.\n"
     "4. Only use information from the provided context - do not use outside knowledge.\n"
     "5. Summarize long information, ideally in bullets where needed.\n"
     "6. If the information is not in the context, say so politely.\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


@st.cache_resource(show_spinner="Reading and indexing the PDF...")
def build_vector_store(file_bytes: bytes):
    """Parse, chunk and embed a PDF once. Cached, so asking more questions
    about the same file does not re-embed it."""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        text = "\n".join((page.extract_text() or "") for page in pdf.pages)

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_text(text)
    if not chunks:
        return None  # e.g. a scanned PDF with no extractable text

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    return FAISS.from_texts(chunks, embeddings)


st.set_page_config(page_title="PDF Q&A Chatbot (RAG)")
st.header("PDF Question-Answering Chatbot")

if not os.getenv("GOOGLE_API_KEY"):
    st.error("GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key.")
    st.stop()

with st.sidebar:
    st.title("Your document")
    file = st.file_uploader("Upload a PDF and ask questions about it", type="pdf")

if file is None:
    st.info("Upload a PDF in the sidebar to get started.")
    st.stop()

vector_store = build_vector_store(file.getvalue())
if vector_store is None:
    st.warning("No readable text found in this PDF (it may be a scanned image).")
    st.stop()

question = st.text_input("Ask your question here")

if question:
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 4})
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0.3, max_output_tokens=2000)
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    try:
        st.write(chain.invoke(question))
    except Exception as exc:  # surface API/model errors instead of a raw traceback
        st.error(f"Could not generate an answer: {exc}")
