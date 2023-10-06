import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import textwrap

# Set Hugging Face API token (if not already set)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Load pre-trained embeddings
embeddings = HuggingFaceEmbeddings()

# Load FAISS vector store (you can create a new one if needed)
db = FAISS.load("faiss_index")

def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

def load_and_process_document(file_path):
    # Load and split the document
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()

    # Split the pages into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(pages)

    return docs

def search_question(query):
    # Search for the most relevant page based on the query
    docs = db.similarity_search(query)
    if docs:
        return wrap_text_preserve_newlines(docs[0].page_content)
    else:
        return "No matching content found."

# Streamlit UI
st.title("PDF Question-Answering Chatbot")
st.sidebar.title("Upload a Document")

uploaded_file = st.sidebar.file_uploader("Upload a PDF Document", type=["pdf"])

if uploaded_file:
    st.sidebar.markdown("### Enter your question:")
    user_question = st.sidebar.text_input("Ask your question here:")

    if st.sidebar.button("Search"):
        st.subheader("Document Content:")
        docs = load_and_process_document(uploaded_file)
        for doc in docs:
            st.write(wrap_text_preserve_newlines(doc.page_content))
        st.subheader("Answer:")
        answer = search_question(user_question)
        st.write(answer)

st.sidebar.text("Note: Please make sure the PDF document format is supported by the chatbot.")

st.markdown("Powered by Langchain and Hugging Face Transformers.")
