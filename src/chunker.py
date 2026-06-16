from tqdm import tqdm
from pathlib import Path
from typing import Generator, List, Dict, Any
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import config


class RecursiveChunker:
    def __init__(self):
        self.chunk_size = config.chunking.CHUNK_SIZE
        self.chunk_overlap = config.chunking.CHUNK_OVERLAP
        self.pdf_dir = config.paths.pdf_dir
        self.descriptions_dir = config.paths.descriptions_dir

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def _get_list_filename(self, dir_path: Path, extension: str) -> List[Path]:
        ext = extension.lstrip(".")
        return list(dir_path.rglob(f"*.{ext}"))

    def split_docs(self, count_chunk_in_batch: int = 10) -> Generator[List[Dict[str, Any]], None, None]:
        list_filename_pdf = self._get_list_filename(self.pdf_dir, "pdf")
        list_description = self._get_list_filename(self.descriptions_dir, "txt")

        if not list_filename_pdf:
            return

        current_batch = []

        for filename in tqdm(list_filename_pdf):
            loader = PyMuPDFLoader(str(filename))
            pdf_stem = filename.stem

            for page in loader.lazy_load():
                docs = self._splitter.split_documents([page])
                page_num = page.metadata.get("page", 0) + 1

                if not docs:
                    continue

                img_prefix = f"{pdf_stem}_p{page_num}_img"
                current_page_descs = [f for f in list_description if f.name.startswith(img_prefix)]

                page_image_context = ""
                if current_page_descs:
                    descriptions_texts = []
                    for desc_path in sorted(current_page_descs):
                        try:
                            descriptions_texts.append(desc_path.read_text(encoding="utf-8").strip())
                        except Exception as e:
                            print(f"Ошибка чтения описания {desc_path.name}: {e}")

                    if descriptions_texts:
                        page_image_context = "\n\n[Иллюстрации на этой странице]:\n" + "\n".join(descriptions_texts)

                for index, doc in enumerate(docs):
                    final_text = doc.page_content
                    if page_image_context:
                        final_text += page_image_context

                    current_batch.append({
                        "text": final_text,
                        "metadata": {
                            "source_file": filename.name,
                            "page": page_num,
                            "chunk_index": index,
                            "has_images": bool(page_image_context),
                            "descriptions": current_page_descs
                        }
                    })

                    if len(current_batch) >= count_chunk_in_batch:
                        yield current_batch
                        current_batch = []

        if current_batch:
            yield current_batch