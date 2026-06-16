import logging
import requests
from prometheus_client import start_http_server, Counter, Histogram

class HTTPElasticHandler(logging.Handler):
    """Кастомный надежный хэндлер для прямой и мгновенной отправки логов в Elasticsearch."""
    def __init__(self, host="127.0.0.1", port=9200, index_name="python-logs"):
        super().__init__()
        self.url = f"http://{host}:{port}/{index_name}/_doc"

    def emit(self, record):
        log_entry = {
            "@timestamp": logging.Formatter().formatTime(record, "%Y-%m-%dT%H:%M:%S.000Z"),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "filename": record.filename,
            "line_number": record.lineno
        }
        
        if record.exc_info:
            log_entry["exception"] = logging.Formatter().formatException(record.exc_info)

        try:
            response = requests.post(self.url, json=log_entry, timeout=5)
            # ЖЕЛЕЗНО ИСПРАВЛЕНО: Простая проверка без багов синтаксиса
            if response.status_code == 200 or response.status_code == 201:
                print(f"📡 [Elasticsearch] Лог успешно отправлен! (Код: {response.status_code})")
            else:
                print(f"⚠️ [Elasticsearch] База вернула странный статус: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ [Elasticsearch] Ошибка сети при отправке лога: {e}")


class AppMonitor:
    def __init__(self):
        # Метрики для Prometheus
        self.query_counter = Counter("rag_user_queries_total", "Общее количество запросов")
        self.query_latency = Histogram("rag_query_latency_seconds", "Время обработки вопроса")
        try:
            start_http_server(8000)
        except OSError:
            pass

        # Настройка логов для Kibana
        self.logger = logging.getLogger("kibana_logger")
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            es_handler = HTTPElasticHandler()
            self.logger.addHandler(es_handler)
            
            # Консольный хэндлер
            console = logging.StreamHandler()
            console.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            self.logger.addHandler(console)

monitor = AppMonitor()
