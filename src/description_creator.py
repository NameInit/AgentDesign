from tqdm import tqdm
from pathlib import Path

from .config import config
from .ollama_vision_describer import OllamaVisionDescriber


class DescriptionCreator:
    def __init__(self):
        self.vision_model = config.ollama.VISION_MODEL
        self.vision_prompt = config.ollama.PROMPT_VISION
        self.images_dir = config.paths.images_dir
        self.descripions_dir = config.paths.descriptions_dir
        self.describer = OllamaVisionDescriber()
        self.prompt_chain = False

    def create_descriptions(self):
        images_list = list(self.images_dir.glob("*"))
        
        if not len(images_list):
            return

        for image_path in tqdm(images_list):
            description_path = self.descripions_dir / f"{image_path.stem}.txt"
            if description_path.exists():
                continue
            description = self._create_description_from_one_image(image_path)
            self._save_description(description_path, description)

        return
    
    def _create_description_from_one_image(self, image_path: Path) -> str:
        try:
            with open(image_path, "rb") as image:
                image_bytes = image.read()
            return self.describer.get_describe_image(image_bytes)
            
        except Exception as e:
            print(f"Ошибка открытия файла {image_path.name}: {e}")
            return None
    
    def _create_description_from_one_image_prompt_chain() -> str:
        return ""

    def _save_description(self, description_path:Path, description:str):
        try:            
            with open(description_path, "w", encoding="utf-8") as f:
                f.write(description)
                
        except Exception as e:
            print(f"\nКритическая ошибка при записи файла {description_path.name}: {e}")

    def clear_descripions(self):
        if not self.descripions_dir.exists():
            print(f"Директория {self.descripions_dir} не существует. Очистка не требуется.")
            return

        descripions_files = list(self.descripions_dir.iterdir())

        for descripions_path in tqdm(descripions_files, desc="Удаление файлов описаний изображений"):
            try:
                descripions_path.unlink()
            except Exception as e:
                print(f"Не удалось удалить файл {descripions_path.name}: {e}")

        return