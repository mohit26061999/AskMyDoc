import streamlit as st
import tempfile
from unstructured.partition.auto import partition
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.text_splitter import TokenTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# ----------- Function to extract text -----------
def extract_text(file) -> str:
    # Save uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.name) as tmp:
        tmp.write(file.getbuffer())
        tmp_path = tmp.name

    elements = partition(filename=tmp_path)
    return "\n".join([el.text for el in elements if el.text])

# ----------- Streamlit UI -----------
st.set_page_config(page_title="Document Q&A", page_icon="📄", layout="wide")
st.title("📄 Document Q&A with RAG")
st.write("Upload a PDF, Word, or TXT file and chat with it as many times as you want.")

uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "doc", "txt"])

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded_file is not None and st.session_state.vectorstore is None:
    with st.spinner("Extracting text..."):
        text = extract_text(uploaded_file)

    st.subheader("Extracted Text Preview:")
    st.text_area("Output", text[:2000], height=200)  # show first 2000 chars

    # Split into chunks
    splitter = TokenTextSplitter(
    chunk_size=512,  
    chunk_overlap=50
    )
    chunks = splitter.create_documents([text])

    with st.spinner("Generating embeddings..."):
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)

# If we have a vectorstore ready, enable chat
if st.session_state.vectorstore is not None:
    retriever = st.session_state.vectorstore.as_retriever(search_type='similarity', search_kwargs={"k": 4})
    llm = OllamaLLM(model='llama3', temperature=0)

    # Prompt template
    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer only from the provided context below.
        If the context is insufficient, just say "I don't know."

        Context:
        {context}

        Question: {question}
        """,
        input_variables=['context', 'question']
    )

    # Sidebar reset option
    with st.sidebar:
        if st.button("🧹 Clear Chat"):
            st.session_state.chat_history = []
            st.success("Chat history cleared!")

    # Chat interface
    st.subheader("💬 Chat with your document")

    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(chat["question"])
        with st.chat_message("assistant"):
            st.markdown(chat["answer"])

    # User input (can ask unlimited times)
    user_question = st.chat_input("Ask a question about the document...")
    if user_question:
        # Show user message immediately
        st.chat_message("user").markdown(user_question)

        with st.spinner("Retrieving context..."):
            retrieved_docs = retriever.invoke(user_question)
            context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

        final_prompt = prompt.format(context=context_text, question=user_question)

        with st.spinner("Generating answer..."):
            result = llm.invoke(final_prompt)

        # Show assistant response
        with st.chat_message("assistant"):
            st.markdown(result)

        # Save to history
        st.session_state.chat_history.append({"question": user_question, "answer": result})
