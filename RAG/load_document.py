from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pathlib import Path


pdf_files = [
    "Deployment.pdf", "another_document.pdf", "doccker.pdf"
]

documents = []
for pdf_file in pdf_files:
    pdf_path = Path(__file__).parent / pdf_file
    documents.extend(PyPDFLoader(str(pdf_path)).load())


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

chunks = text_splitter.split_documents(documents)
print(f"Total document chunks created: {len(chunks)}")


from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

embedding_model = OllamaEmbeddings(model="qwen3-embedding:0.6b")

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)

print("Vector database created and persisted locally.")