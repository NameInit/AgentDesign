import tqdm
import fitz
import json
import typing
import base64
import requests
from src.config import config
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter


class OllamaVisionHelper:
    def __init__(self):
        self.model_name = config.ollama.VISION_MODEL
        self.api_url = f"{config.ollama.BASE_URL}/api/generate"
        self.timeout = config.ollama.TIMEOUT

    def describe_image(self, image_path: Path):
        try:
            with open(image_path, "rb") as image_file:
                encoded_image: str = base64.b64encode(image_file.read()).decode('utf-8')
            
            prompt: str = (
                "You are an expert engineering and CAD software assistant (specifically Siemens NX). "
                "Analyze this image from a technical document and describe it precisely based on its type:\n\n"
                
                "1. Identify the TYPE of image (e.g., 3D CAD model, 2D technical drawing, handwritten sketch, "
                "mathematical formula, or Siemens NX 10 user interface screenshot).\n"
                
                "2. Technical content description:\n"
                "   - If it is a 3D PART/DRAWING: Describe shapes, geometry, holes, threads, views, and visible labels.\n"
                "   - If it is a FORMULA: Transcribe the variables and symbols, explain what components are visible.\n"
                "   - If it is an NX 10 INTERFACE screenshot: Name the visible dialog boxes, buttons, menus, paths, "
                "     or CAD features highlighted (e.g., Extrude, Blend, Sketcher, Part Navigator).\n"
                "   - If it is HANDWRITTEN text/sketch: Transcribe the text and describe the rough drawing or dimensions.\n\n"
                
                "List all visible labels, dimensions, text, values, or button names. Be concise, technical, and accurate."
            )
            
            payload: dict[str, typing.Any] = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [encoded_image],
                "stream": False
            }
            
            response: requests.Response = requests.post(
                self.api_url, 
                json=payload, 
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return response.json().get("response", "").strip()
            
            print(f"⚠ Ollama вернула код {response.status_code} для файла {image_path.name}")
            return ""

        except Exception as e:
            print(f"❌ Ошибка Ollama Vision при анализе картинки {image_path.name}: {e}")
            return ""

class DocumentChunker:
    def __init__(self):
        self.input_dir = config.paths.INPUT_DIR
        self.images_dir = config.paths.images_dir
        self.output_json = config.paths.chunks_json_path
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunking.CHUNK_SIZE,
            chunk_overlap=config.chunking.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " "]
        )
        self.vision_helper = OllamaVisionHelper()

    def run(self):
        all_chunks = list()
        pdf_files = list(self.input_dir.rglob("*.pdf"))
        self.extracted_images_registry = set()

        print(f"All files pdf = {len(pdf_files)}")

        if not pdf_files or len(pdf_files)==0:
            return []
        
        for pdf_path in tqdm.tqdm(pdf_files):
            pdf_chunks = self._parse_pdf(pdf_path)
            all_chunks.extend(pdf_chunks)
            
        with open(self.output_json, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=4)
            print(f"✅ Чанки успешно сохранены в {self.output_json}")

        return all_chunks

    def _parse_pdf(self, path):
        doc = fitz.open(path)
        pdf_chunks = list()

        for i, page in enumerate(doc):
            text = page.get_text()
            image_descriptions = self._extract_image_on_page(page,i+1,path)
            page_text_chunks = self.text_splitter.split_text(text)

            for chunk_text in page_text_chunks:
                enriched_text = chunk_text

                if image_descriptions:
                    enriched_text += "\n\n[Контекст изображений на этой странице]:\n"
                    for img_data in image_descriptions:
                        enriched_text += f"(Описание детали на чертеже: {img_data['description']})\n"
                    
                chunk_struct = {
                    "text": enriched_text,
                    "metadata": {
                        "source_file": path.name,
                        "sub_folder": path.parent.name,
                        "page": i+1,
                        "images": [img["path"] for img in image_descriptions],
                        "chunk_raw_length": len(chunk_text),
                        "chunk_enriched_length": len(enriched_text)
                    }
                }
                pdf_chunks.append(chunk_struct)
        
        doc.close()

        return pdf_chunks
    
    def _extract_image_on_page(self, page: fitz.Page, page_num:int, pdf_path:Path):
        descriptions = list()
        image_info_list = page.get_image_info(xrefs=True)
        
        for img_info in image_info_list:
            xref = img_info.get("xref")

            if not xref or xref==0 or (pdf_path, xref) in self.extracted_images_registry:
                continue
            
            self.extracted_images_registry.add((pdf_path, xref))

            base_image = page.parent.extract_image(xref)
            if not base_image:
                continue

            image_bytes: bytes = base_image["image"]
            image_ext: str = base_image["ext"]

            img_name: str = f"{pdf_path.stem}_p{page_num}_img_{xref}.{image_ext}"
            img_path: Path = self.images_dir / img_name

            with open(img_path, "wb") as f:
                f.write(image_bytes)

            description = self.vision_helper.describe_image(img_path)

            descriptions.append({
                "path": str(img_path.resolve()),
                "description": description
            })

        return descriptions