
# PDF Question Answering Chatbot
This repository contains code for a PDF question answering chatbot that can extract information and answer questions from PDF documents. The chatbot uses various natural language processing and machine learning techniques to understand and respond to questions based on the content of PDF files.


## Getting Started
```bash
pip install langchain
pip install huggingface_hub
pip install sentence_transformers
pip install pypdf
pip install faiss-cpu
```
## Usage/Examples
Usage
To use the chatbot, follow these steps:

Set your Hugging Face Hub API token as an environment variable:

```javascript
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "xxxxxxxx"

```
Load a PDF document using the PyPDFLoader
```bash
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader('path_to_your_pdf_file.pdf')
pages = loader.load_and_split()
```
Split the document into smaller text chunks for processing:
```bash
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(pages)
```
Create embeddings for the text chunks:
```bash
from langchain.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings()
```
Build a vector store using FAISS:
```bash
from langchain.vectorstores import FAISS
db = FAISS.from_documents(docs, embeddings)
```
```bash
query = "Your question here"
results = db.similarity_search(query)
```




