from src.config import config
from src.parser import DocumentChunker


if __name__ == "__main__":
    document_chunker = DocumentChunker()
    document_chunker.run()