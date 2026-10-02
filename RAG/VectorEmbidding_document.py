from langchain_ollama import OllamaEmbedding
from langchain_chroma import Chroma

embedding_model = OllamaEmbedding(model="qwen3-embedding:0.6b")

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)

print("Vector database created and persisted locally.")