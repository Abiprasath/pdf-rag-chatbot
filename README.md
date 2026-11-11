# PDF Question-Answering Chatbot (RAG)

Upload a PDF and ask questions about it. Answers are generated only from the document's content using Retrieval-Augmented Generation (RAG).

## How it works

1. **Parse**: extract text from the uploaded PDF with `pdfplumber`.
2. **Chunk**: split into overlapping chunks (1000 characters, 200 overlap) with LangChain's `RecursiveCharacterTextSplitter`.
3. **Embed and index**: create embeddings with Gemini (`gemini-embedding-001`) and store them in a FAISS vector index. The index is cached, so each PDF is embedded once per session.
4. **Retrieve**: fetch the 4 most relevant chunks using MMR (Maximal Marginal Relevance) to reduce redundant context.
5. **Answer**: a context-restricted prompt tells the Gemini model to answer only from the retrieved text and say so when the answer is not in the document, which keeps responses grounded and reduces hallucination.

## Tech stack

Python, Streamlit, LangChain, FAISS, Google Gemini API, pdfplumber

## Setup

```bash
git clone https://github.com/Abiprasath/pdf-rag-chatbot.git
cd pdf-rag-chatbot
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file (see `.env.example`) containing your key from Google AI Studio:

```
GOOGLE_API_KEY=your_gemini_api_key_here
```

Run the app:

```bash
streamlit run app.py
```

## Notes and limitations

- Scanned PDFs without a text layer are not supported (no OCR).
- The API key is read from the environment and is never stored in the code. `.env` is git-ignored.
- The chat model name can be changed with the `GEMINI_MODEL` environment variable.

## Possible improvements

- Source citations (page numbers) with each answer
- Chat history and multi-document support
- A FastAPI backend and Docker image
