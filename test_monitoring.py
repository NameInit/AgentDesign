import time
import random
from src.monitoring import monitor

def run_monitoring_test():
    print("=" * 80)
    print("🧪 СКРИПТ ТЕСТИРОВАНИЯ МОНИТОРИНГА И ЛОГИРОВАНИЯ ЗАПУЩЕН")
    print("=" * 80)
    print("Prometheus метрики доступны по адресу: http://localhost:8000/metrics")
    print("Логи отправляются на Logstash/Kibana через порт 5000")
    print("Для остановки теста нажмите Ctrl + C\n")
    print("-" * 80)

    # Список фейковых вопросов для симуляции нагрузки
    sample_queries = [
        "как выглядит формула Безье?",
        "чертеж сборки редуктора на странице 5",
        "какие допуски у вала?",
        "инструкция по сборке детали",
        "спецификация крепежных болтов"
    ]

    iteration = 1
    while True:
        try:
            # 1. Выбираем случайный технический вопрос
            query = random.choice(sample_queries)
            print(f"\n[Итерация {iteration}] Симулируем запрос пользователя: '{query}'")

            # 2. Увеличиваем счетчик запросов для Prometheus
            monitor.query_counter.inc()

            # 3. Замеряем время обработки запроса с помощью гистограммы Prometheus
            # Конструкция with автоматически отправит время выполнения блока в Grafana
            with monitor.query_latency.time():
                # Симулируем задержку ответа ChromaDB и Ollama (от 0.2 до 2.5 секунд)
                fake_processing_time = random.uniform(0.2, 2.5)
                time.sleep(fake_processing_time)

                # Имитируем случайную ошибку в 20% случаев (например, таймаут Ollama)
                if random.random() < 0.2:
                    raise ConnectionError("Ollama API временно недоступно (Timeout 504)")

            # 4. Если все прошло успешно — отправляем информационный лог для Kibana
            monitor.logger.info(
                f"Запрос '{query}' успешно обработан за {fake_processing_time:.2f} сек."
            )
            print(f"✔️ Успешно залогировано. Время: {fake_processing_time:.2f} сек.")

        except Exception as e:
            # 5. При сбое отправляем лог ошибки с полной цепочкой вызовов (exc_info=True)
            # Именно эти данные Kibana разложит на понятные графики сбоев
            monitor.logger.error(
                f"Ошибка при обработке запроса '{query}': {str(e)}", 
                exc_info=True
            )
            print(f"❌ Зафиксирована ошибка: {e}")

        # Пауза между запросами пользователей (1-3 секунды)
        time.sleep(random.uniform(1.0, 3.0))
        iteration += 1

if __name__ == "__main__":
    try:
        run_monitoring_test()
    except KeyboardInterrupt:
        print("\n👋 Тестирование завершено пользователем.")
