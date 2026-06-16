from src.chunker import RecursiveChunker
from src.chroma_manager import ChromaManager
from src.config import config

if __name__ == "__main__":
    chunker = RecursiveChunker()
    chroma_manager = ChromaManager(config.chunking.COLLECTION_NAME)

    print("Chunking start.")

    for batch in chunker.split_docs(count_chunk_in_batch=50):
        chroma_manager.add_batch(batch)

    print("Chunking finish!")