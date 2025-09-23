from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma  
from langchain_huggingface import HuggingFaceEmbeddings  # updated import
from langchain_community.document_loaders import Docx2txtLoader


def create_vector_db(file_path, persist_directory="db"):
    # 1. Load the Word document
    loader = Docx2txtLoader(file_path)
    documents = loader.load()   # returns a list of Document objects

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = splitter.split_documents(documents)

    # 3. Create embeddings
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # now from langchain_huggingface

    # 4. Store in Chroma
    vector_store = Chroma.from_documents(docs, embedding, persist_directory=persist_directory)
    vector_store.persist()

    return vector_store


db = create_vector_db(r"C:\Users\hp\Desktop\lang_chain\vector_db\backend_doc.docx")
print("✅ Vector DB created and persisted!", db.similarity_search("what is the different endpoint you are using"))
