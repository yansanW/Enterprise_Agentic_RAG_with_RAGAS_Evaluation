# src/inspect_db.py
import os
from src.database import initialize_vectorstore

def inspect_my_vault():
    # 1. Mount the existing database folder from disk
    vectorstore = initialize_vectorstore()
    
    # 2. Extract all raw contents from Chroma
    db_data = vectorstore.get()
    
    total_chunks = len(db_data.get("ids", []))
    metadatas = db_data.get("metadatas", [])
    documents = db_data.get("documents", [])
    
    print("\n=======================================================")
    print("🔍 CHROMADB STORAGE MATRIX AUDIT")
    print("=======================================================")
    print(f"📊 Total Stored Chunks Found : {total_chunks}")
    
    if total_chunks == 0:
        print("⚠️ Your database is currently completely EMPTY.")
        print("=======================================================\n")
        return

    # Sets to gather unique tracking information across all chunks
    unique_sources = set()
    ingestion_timestamps = set()
    parser_strategies = set()

    for meta in metadatas:
        if meta:
            if "source" in meta:
                unique_sources.add(meta["source"])
            if "ingested_at" in meta:
                ingestion_timestamps.add(meta["ingested_at"])
            if "parser_strategy" in meta:
                parser_strategies.add(meta["parser_strategy"])
            
    print("\n📂 Source Files Tracked Inside the Vault:")
    for source in unique_sources:
        print(f"  🔹 {source}")
        
    print("\n⏱️ Ingestion Run Timestamps Found:")
    if ingestion_timestamps:
        for ts in sorted(ingestion_timestamps):
            print(f"  ⏰ {ts}")
    else:
        print("  ⚠️ No 'ingested_at' timestamps found. (Legacy chunk data before upgrade)")

    print("\n⚙️ Parser Execution Strategies Used:")
    if parser_strategies:
        for strategy in parser_strategies:
            print(f"  🛠️ {strategy}")
    else:
        print("  ⚠️ No 'parser_strategy' metadata found.")
        
    print("\n📝 Sample Snippet from Vector Space:")
    print(f'  "{documents[0][:150]}..."')
    print("=======================================================\n")

if __name__ == "__main__":
    inspect_my_vault()