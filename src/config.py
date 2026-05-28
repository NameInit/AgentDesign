import os
from dataclasses import dataclass, field
from pathlib import Path

@dataclass(frozen=True)
class OllamaConfig:
    BASE_URL: str = "http://localhost:11434"
    VISION_MODEL: str = "llava:7b"
    EMBEDDING_MODEL: str = "nomic-embed-text"
    LLM_MODEL: str = "llava:7b"
    TIMEOUT: int = 120

@dataclass(frozen=True)
class ChunkingConfig:
    CHUNK_SIZE: int = 1500
    CHUNK_OVERLAP: int = 300
    IMAGE_DPI: int = 150

@dataclass(frozen=True)
class PathConfig:
    BASE_DIR: Path = Path(__file__).resolve().parent
    
    INPUT_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "documents" / "src")
    OUTPUT_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    
    @property
    def images_dir(self) -> Path:
        return self.OUTPUT_DIR / "documents" / "extracted_images"
    
    @property
    def chunks_json_path(self) -> Path:
        return self.OUTPUT_DIR / "db" / "chunks.json"
    
    @property
    def vector_db_dir(self) -> Path:
        return self.OUTPUT_DIR / "chroma_db"

    def create_directories(self) -> None:
        self.INPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)


class AppConfig:
    ollama: OllamaConfig = OllamaConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    paths: PathConfig = PathConfig()

    @classmethod
    def initialize(cls):
        cls.paths.create_directories()

config = AppConfig()
