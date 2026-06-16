import chromadb
from chromadb.config import Settings
import requests
from .config import config

class RAGQueryEngine:
    def __init__(self, collection_name: str = "pdf_documents"):
        # Жестко отключаем телеметрию ChromaDB при создании клиента
        self.client = chromadb.PersistentClient(
            path=str(config.paths.chroma_db_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Получаем чистую коллекцию БЕЗ привязки сбойной функции эмбеддингов ChromaDB
        self.collection = self.client.get_collection(name=collection_name)
        
        self.ollama_emb_url = "http://localhost:11434/api/embeddings"
        self.ollama_chat_url = "http://localhost:11434/api/chat"

    def _get_query_embedding(self, text: str) -> list:
        """Получает вектор для вопроса пользователя напрямую из API Ollama."""
        try:
            response = requests.post(
                self.ollama_emb_url,
                json={
                    "model": "mxbai-embed-large",
                    "prompt": text,
                    "options": {"num_ctx": 2048}
                },
                timeout=15
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            print(f"\n[Ошибка векторизации вопроса]: {e}")
            return None

    def query(self, user_question: str, n_results: int = 3) -> str:
        """Ищет релевантный контекст по готовому вектору и генерирует ответ через Qwen."""
        
        # 1. Получаем вектор вопроса вручную
        query_vector = self._get_query_embedding(user_question)
        if not query_vector:
            return "Не удалось преобразовать вопрос в вектор. Проверьте, запущена ли Ollama."

        # 2. Делаем поисковый запрос в базу строго по ВЕКТОРУ (query_embeddings)
        # Так как текстовый query_texts здесь не передается, ChromaDB не будет запускать
        # свою сбойную телеметрию CollectionQueryEvent и не зависнет!
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=n_results
        )

        if not results or "documents" not in results or not results["documents"] or not results["documents"][0]:
            return "К сожалению, в вашей базе знаний нет информации по этому вопросу."

        # Распаковываем результаты поиска (Chroma возвращает список списков)
        retrieved_chunks = results["documents"][0]
        metadatas = results["metadatas"][0] if results["metadatas"] else []

        # 3. Собираем контекст из документов для отправки в модель
        context_blocks = []
        for text, meta in zip(retrieved_chunks, metadatas):
            source = meta.get("source_file", "Неизвестный файл")
            page = meta.get("page", "?")
            context_blocks.append(f"--- Источник: {source} (Стр. {page}) ---\n{text}")
            
        full_context = "\n\n".join(context_blocks)

        # 4. Формируем системную инструкцию
        system_instruction = (
            "Ты — квалифицированный инженерный ассистент. Твоя задача — ответить на вопрос пользователя строго на основе предоставленного контекста.\n"
            "Если в контексте есть математические формулы, зависимости или технические параметры, воспроизведи их максимально точно.\n"
            "Если в предоставленном контексте нет ответа на вопрос, честно скажи: 'В предоставленных документах нет информации по данному вопросу.' Не придумывай ничего от себя."
        )
        
        user_message_content = f"КОНТЕКСТ ИЗ ВАШИХ PDF-ДОКУМЕНТОВ:\n{full_context}\n\nВопрос: {user_question}"

        # 5. Генерируем финальный ответ через эндпоинт /api/chat
        try:
            response = requests.post(
                self.ollama_chat_url,
                json={
                    "model": "qwen2.5:3b",
                    "messages": [
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_message_content}
                    ],
                    "stream": False,
                    "options": {
                        "num_ctx": 8192
                    }
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
        except Exception as e:
            return f"Ошибка при генерации ответа моделью Qwen: {e}"
