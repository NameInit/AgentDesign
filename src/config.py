import os
from dataclasses import dataclass, field
from pathlib import Path

@dataclass(frozen=True)
class OllamaConfig:
    BASE_URL: str = "http://localhost:11434"
    VISION_MODEL: str = "qwen2.5vl:7b-q4_K_M"
    # VISION_MODEL: str = "qwen2.5vl:3b"
    EMBEDDING_MODEL: str = "nomic-embed-text"
    LLM_MODEL: str = "qwen2.5vl:7b-q4_K_M"
    TIMEOUT: int = 120
    PROMPT_VISION = """
Ты — эксперт в области инженерной геометрии, САПР и компьютерной графики. Проведи строгий технический анализ изображения из курса лекций. Отвечай на русском.

Структура ответа:
1. КЛАССИФИКАЦИЯ: Точный тип (математическая формула, 2D/3D геометрическая фигура, чертеж проекций, схема/алгоритм, график функции, интерфейс Siemens NX).
2. СУТЬ И КОНТЕКСТ: Инженерный и геометрический смысл. Что за объект, уравнение или закон здесь изображены? (Анализируй контекст: это могут быть аналитические кривые/поверхности, полигональные сетки, твердотельные операции, сопряжения, матрицы преобразований или методы начертательной геометрии).
3. ДЕТАЛИЗАЦИЯ: Если формула — переведи в LaTeX. Если фигура/чертеж/схема — перечисли все ключевые элементы (опорные точки, векторы, ребра, грани, оси, размеры, геометрические ограничения).

Пиши строго фактами по изображению, называй объекты их точными научными терминами. Используй списки.
"""
    
# """
# Ты — эксперт в области геометрического моделирования, САПР (CAD), инженерной графики и компьютерного зрения.
# Твоя задача — провести глубокий технический анализ изображения (схемы, формулы, чертежа или интерфейса Siemens NX).

# Выполни анализ строго по следующим шагам:

# 1. КЛАССИФИКАЦИЯ: Определи тип изображения (например: математическая формула, 2D-эскиз/скетч, 3D-деталь, схема/диаграмма, элемент интерфейса Siemens NX, графическое представление кривой/поверхности).
# 2. СУТЬ И КОНТЕКСТ: Опиши технический смысл изображения. Что конкретно здесь происходит или изображено? (Например: "Показано редактирование кубической кривой Безье с помощью четырех контрольных точек в CAD-системе").
# 3. ДЕТАЛИЗАЦИЯ И СИМВОЛЫ: 
#    - Если это формула: переведи её в LaTeX и распиши переменные.
#    - Если это эскиз, деталь или интерфейс NX: перечисли ключевые элементы (опорные точки, векторы касательных, оси координат, элементы сопряжения, геометрические ограничения/размеры, стрелки). Укажи их цвета и назначение.
# 4. ОГРАНИЧЕНИЕ ФАНТАЗИИ: Описывай только то, что явно присутствует на изображении. Не придумывай физический контекст (диффузия, время, термодинамика), если это геометрия или CAD. Если элемент интерфейса или символ неизвестен, пиши: "Элемент неопознан".

# Отвечай строго на русском языке, структурировано (используй списки и жирный шрифт для ключевых терминов).
# """

@dataclass(frozen=True)
class ChunkingConfig:
    CHUNK_SIZE: int = 1500
    CHUNK_OVERLAP: int = 300
    IMAGE_DPI: int = 150

@dataclass(frozen=True)
class PathConfig:
    PROJECT_DIR: Path = Path(__file__).resolve().parent.parent
    
    @property
    def pdf_dir(self) -> Path:
        return self.PROJECT_DIR / "documents" / "src"

    @property
    def images_dir(self) -> Path:
        return self.PROJECT_DIR / "documents" / "extracted_images" / "images"
    
    @property
    def descriptions_dir(self) -> Path:
        return self.PROJECT_DIR / "documents" / "extracted_images" / "descriptions"

    @property
    def chunks_json_path(self) -> Path:
        return self.PROJECT_DIR / "db" / "chunks.json"
    
    @property
    def vector_db_dir(self) -> Path:
        return self.PROJECT_DIR / "db" / "chroma_db"

class AppConfig:
    ollama: OllamaConfig = OllamaConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    paths: PathConfig = PathConfig()

    @classmethod
    def initialize(cls):
        cls.paths.create_directories()

config = AppConfig()
