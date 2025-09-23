from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def main():
    # initialize embedding llm
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # load already persisted db
    vector_store = Chroma(persist_directory="db", embedding_function=embedding)
    print("✅ Vector DB loaded successfully!")
    
    # interact the db
    print("\nType your query (or 'exit' to quit):")
    while True:
        query = input(">>> ")
        if query.lower() in ["exit", "quit"]:
            print("Exiting...")
            break
        
        result = vector_store.similarity_search(query, k=3)
        
        if not result:
            print("no relevent info")
            continue
        
        for i, doc in enumerate(result):
            print(f"\n--- Result {i+1} ---")
            print(doc.page_content)
        
if __name__ == "__main__":
    main()
        