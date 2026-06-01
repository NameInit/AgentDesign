from src.ollama_vision_describer import OllamaVisionDescriber
from src.config import config

if __name__ == "__main__":
    describer = OllamaVisionDescriber()

    with open("./documents/extracted_images/images/001_Введение к курсу лекций по дисциплине Геом моделирование_p5_img1.jpeg", "rb") as f:
        bytes = f.read()

    print("Generation...")
    print(describer.get_describe_image(bytes))