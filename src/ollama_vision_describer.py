import ollama

from .config import config


class OllamaVisionDescriber:
    def __init__(self):
        self.model_name = config.ollama.VISION_MODEL
        self.prompt = config.ollama.PROMPT_VISION

    def get_describe_image(self, image_bytes) -> str:
        response = ollama.chat(
            model = self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": self.prompt,
                    "images": [image_bytes],
                }
            ],
            keep_alive=-1
        )
        return response['message']['content'].strip()