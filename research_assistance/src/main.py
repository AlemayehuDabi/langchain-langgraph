import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader

# -----------------------------
# Embeddings
# -----------------------------
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# -----------------------------
# Initialize or reconnect to DB
# -----------------------------
def initialize_db(embed, persist_dir="chroma_db"):
    db = Chroma(
        persist_directory=persist_dir,
        embedding_function=embed
    )
    return db

# -----------------------------
# Load publications
# -----------------------------
def load_publications(folder="publications_md"):
    loader = DirectoryLoader(folder, glob="*.md")
    documents = loader.load()
    return documents

# -----------------------------
# Chunk publications
# -----------------------------
def chunk_publications(documents, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(documents)
    return chunks

# -----------------------------
# Store chunks in vector DB
# -----------------------------
def store_in_vector_db(chunks, embed, persist_dir="chroma_db"):
    db = Chroma.from_documents(
        chunks, embedding_function=embed, persist_directory=persist_dir
    )
    db.persist()
    return db

# -----------------------------
# Main function
# -----------------------------
def main():
    persist_dir = "chroma_db"
    embedding = get_embeddings()

    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        print("📂 DB exists! Reconnecting...")
        db = initialize_db(embedding, persist_dir)
    else:
        print("📚 DB not found. Building from publications...")
        publications = load_publications()
        chunks = chunk_publications(publications)
        db = store_in_vector_db(chunks, embedding, persist_dir)

    print("✅ ChromaDB is ready!")

    # Example query
    query = "What is tensor decomposition?"
    results = db.similarity_search(query, k=3)
    for i, r in enumerate(results, 1):
        print(f"\nResult {i}:\n{r.page_content[:300]}...")

# -----------------------------
if __name__ == "__main__":
    main()
