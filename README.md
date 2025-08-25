# 📄 AskMyDoc – AI-Powered Document Q&A

AskMyDoc is an AI-powered assistant that allows users to upload PDF, Word, or TXT files and chat with them. 
Using Retrieval-Augmented Generation (RAG), FAISS, and Ollama LLMs, it provides accurate, context-aware answers from your documents.

## 🚀 Features
- Upload PDF, DOCX, TXT
- Extract text with Unstructured
- Split text into chunks for embeddings
- Store & retrieve with FAISS
- Ask unlimited questions in a chat interface
- Powered by Ollama LLMs

## 🛠️ Tech Stack
- **Frontend/UI**: Streamlit
- **LLM**: Ollama (LLaMA 3)
- **Vector DB**: FAISS
- **Text Processing**: LangChain + Unstructured

## 📦 Installation
```bash
git clone https://github.com/mohit26061999/AskMyDoc.git
cd AskMyDoc
pip install -r requirements.txt
