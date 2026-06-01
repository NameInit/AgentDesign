import fitz
from tqdm import tqdm
from pathlib import Path

from .config import config

class ImageExtractor:
    def __init__(self):
        self.pdf_dir = config.paths.pdf_dir
        self.images_dir = config.paths.images_dir
    
    def extract(self):
        pdf_files = list(self.pdf_dir.rglob("*.pdf"))

        if not len(pdf_files):
            return

        for pdf_path in tqdm(pdf_files, desc="Извлечение картинок из pdf"):
            self._extract_from_one_pdf(pdf_path)

        return
    
    def _extract_from_one_pdf(self, pdf_path: Path):
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"Ошибка открытия файла {pdf_path.name}: {e}")
            return
        
        pdf_name = pdf_path.stem
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_image_info(xrefs=True)

            for img_idx, img_info in enumerate(image_list):
                xref = img_info.get("xref")

                if not xref:
                    continue

                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]

                    filename_base = f"{pdf_name}_p{page_num + 1}_img{img_idx + 1}"
                    img_save_path = self.images_dir / f"{filename_base}.{image_ext}"

                    if img_save_path.exists():
                        continue

                    with open(img_save_path, "wb") as f:
                        f.write(image_bytes)

                except Exception as e:
                    print(f"Ошибка извлечения картинки индекс {img_idx} на странице {page_num + 1}: {e}")
                    continue
        
        doc.close()
        
        return
    
    def clear_images(self):
        if not self.images_dir.exists():
            print(f"Директория {self.images_dir} не существует. Очистка не требуется.")
            return

        image_files = list(self.images_dir.iterdir())

        for img_path in tqdm(image_files, desc="Удаление файлов изображений"):
            try:
                img_path.unlink()
            except Exception as e:
                print(f"Не удалось удалить файл {img_path.name}: {e}")

        return