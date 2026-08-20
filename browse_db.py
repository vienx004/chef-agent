import json
from chef_agent import config
from chef_agent.llm import GeminiLLM
from chef_agent.rag.chroma_retriever import ChromaRecipeRetriever

def main():
    print("=" * 60)
    print("       ***  LOCAL CHROMADB RECIPE BROWSER  ***       ")
    print("=" * 60)

    print(f"Connecting to local Chroma DB at: {config.CHROMA_DB_PATH}")

    try:
        # Initialize Gemini LLM to satisfy the retriever constructor requirements (not used for simple list)
        llm = GeminiLLM(api_key=config.GEMINI_API_KEY)
        
        retriever = ChromaRecipeRetriever(
            llm=llm,
            db_path=config.CHROMA_DB_PATH,
            seed_json_path=config.RECIPE_DB_PATH
        )
        
        # Get count
        count = retriever.collection.count()
        print(f"Total recipes stored in vector database: {count}")
        print("-" * 60)
        
        if count == 0:
            print("The database is currently empty.")
            return

        # Fetch all items from the collection
        results = retriever.collection.get()
        
        ids = results.get("ids", [])
        metadatas = results.get("metadatas", [])
        
        for idx, (doc_id, meta) in enumerate(zip(ids, metadatas), 1):
            print(f"{idx}. [{doc_id}] - {meta.get('title', 'Untitled')}")
            print(f"   Description: {meta.get('description', '')}")
            
            # Print keywords
            keywords = meta.get("keywords", "")
            if keywords:
                print(f"   Keywords:    {keywords}")
            print("-" * 60)

    except Exception as e:
        print(f"❌ Error browsing local Chroma DB: {e}")
        print("Make sure you have run 'python main.py' at least once to seed the database.")

if __name__ == "__main__":
    main()
