from .config import config
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

class ChromaManager:
    def __init__(self, collection_name = "pdf_documents"):
        self.client = chromadb.PersistentClient(str(config.paths.chroma_db_dir))
        
        self.ollama_ef = OllamaEmbeddingFunction(
            url="http://localhost:11434/api/embeddings",
            model_name="mxbai-embed-large"
        )
        
        self.ollama_ef._options = {"num_ctx": 4096}

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.ollama_ef,
            metadata={"hnsw:space": "cosine"}
        )

    def _generate_id(self, metadata: dict) -> str:
        return f"{metadata.get('source_file', 'unknown')}_page_{metadata.get('page', 0)}_chunk_{metadata.get('chunk_index', 0)}"

    def _sanitize_metadata(self, metadata: dict) -> dict:
        return {
            "source_file": str(metadata.get("source_file", "")),
            "page": int(metadata.get("page", 0)),
            "chunk_index": int(metadata.get("chunk_index", 0)),
            "has_images": bool(metadata.get("has_images", False))
        }

    def add_batch(self, batch: list):
        if not batch:
            return
        
        documents = []
        ids = []
        metadatas = []

        for chunk in batch:
            text = chunk.get("text", "").strip()
            if not text:
                continue
                
            documents.append(text)
            ids.append(self._generate_id(chunk["metadata"]))
            metadatas.append(self._sanitize_metadata(chunk["metadata"]))

        if not documents:
            return

        try:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
        except Exception as e:
            print(f"\n[Ошибка сохранения батча]: {e}")
