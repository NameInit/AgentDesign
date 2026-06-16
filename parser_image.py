from src.image_extractor import ImageExtractor
from src.description_creator import DescriptionCreator

if __name__ == "__main__":
    document_chunker = ImageExtractor()
    document_chunker.extract()

    description_creator = DescriptionCreator()
    description_creator.create_descriptions()