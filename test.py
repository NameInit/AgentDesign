import base64
from pathlib import Path
import importlib.util
from PIL import Image
import io
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole

# 1. Загружаем API ключ из .config/config.py
spec = importlib.util.spec_from_file_location(
    "dot_config", 
    Path(__file__).parent / ".config" / "config.py"
)
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)

# Получаем ключ из вашего конфига
GIGACHAT_CREDENTIALS = config_module.api_key_gigachat
GIGACHAT_SCOPE = getattr(config_module, 'gigachat_scope', 'GIGACHAT_API_PERS')

# 2. Путь к вашему изображению
test_image = Path("/home/ilya/document/extracted_images/001_Введение_к_курсу_лекций_по_дисциплине_Геом_моделирование_p4_img_44.jpeg")

# 3. Инициализация клиента с мультимодальной моделью
client = GigaChat(
    credentials=GIGACHAT_CREDENTIALS,
    scope=GIGACHAT_SCOPE,
    verify_ssl_certs=False,
    model="gigavision"  # Предполагаемая модель, позже проверим точную
)

# 4. Проверка доступных моделей (корректный способ)
print("Проверка доступных моделей:")
models_list = client.aget_models().sync()
for model in models_list.data:
    print(model.name)

# 5. Загружаем изображение в облачное хранилище
print("🔄 Загрузка изображения...")
with open(test_image, "rb") as f:
    uploaded_file = client.upload_file(f, purpose="general")

# Правильное имя атрибута — 'id_'
file_id = uploaded_file.id_
print(f"✅ Изображение загружено, file_id: {file_id}")

# 6. Универсальная подсказка-промпт для описания любого изображения
prompt_text = """Опиши, пожалуйста, что изображено на прикрепленном изображении. Определи ключевые объекты, сцену, обстановку, эмоции персонажей (если есть), общую атмосферу кадра. Дай развернутое описание увиденного."""

# 7. Формирование запроса с прикрепленным изображением
request = Chat(
    messages=[
        Messages(
            role=MessagesRole.USER,
            content=prompt_text,  # Текстовый промпт
            attachments=[file_id] # Прикрепление изображения
        )
    ],
    model="gigavision"  # Модель, выбранная после проверки
)

# 8. Отправка запроса и получение результата
print("🔄 Отправка запроса к GigaChat-Vision...")
response = client.chat(request)

# 9. Выводим полученный ответ
print("\n" + "=" * 60)
print("ОПИСАНИЕ ИЗОБРАЖЕНИЯ:")
print("=" * 60)
print(response.choices[0].message.content)