import os
import sys
import logging

# ИСПРАВЛЕНО: правильное название метода — getLogger
logging.getLogger("chromadb").setLevel(logging.ERROR)
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Дополнительно глушим ворнинги на уровне встроенного механизма Python
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Скрываем системный поток ошибок на время импорта проблемных библиотек
sys.stderr = open(os.devnull, 'w')

from src.query_engine import RAGQueryEngine
from src.config import config

# Возвращаем стандартный поток ошибок в нормальное русло
sys.stderr = sys.__stderr__


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    clear_screen()
    print("=" * 80)
    print("🚀 ИНЖЕНЕРНЫЙ RAG-АССИСТЕНТ НА БАЗЕ QWEN 2.5 & CHROMADB ЗАПУЩЕН")
    print("=" * 80)
    print(f"Коллекция в БД: {config.chunking.COLLECTION_NAME}")
    print("Вы можете задавать любые вопросы по вашим PDF-документам и эскизам.")
    print("Для выхода из чата введите: exit, quit или уйти\n")
    print("-" * 80)

    try:
        query_engine = RAGQueryEngine(collection_name=config.chunking.COLLECTION_NAME)
    except Exception as e:
        print(f"❌ Ошибка подключения к ChromaDB: {e}")
        return

    while True:
        try:
            user_query = input("\n📝 Ваш вопрос: ").strip()
            
            if user_query.lower() in ["exit", "quit", "exit()", "уйти", "выход"]:
                print("\n👋 Сессия завершена. До встречи!")
                break
                
            if not user_query:
                continue

            print("\n🔍 Ищем релевантный контекст и графику...")
            
            # Отправляем запрос в движок
            answer = query_engine.query(user_query, n_results=3)
            
            print("\n🤖 Ответ ассистента:")
            print("-" * 40)
            print(answer)
            print("-" * 40)

        except KeyboardInterrupt:
            print("\n\n👋 Сессия прервана пользователем. До встречи!")
            break
        except Exception as e:
            print(f"\n❌ Произошла непредвиденная ошибка: {e}")

if __name__ == "__main__":
    main()
