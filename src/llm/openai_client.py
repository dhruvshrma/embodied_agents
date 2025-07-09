import os
import openai
from src.llm.base import LLMClient

class OpenAIClient(LLMClient):
    def __init__(self, model: str = "gpt-3.5-turbo", temperature: float = 0.7):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.model = model
        self.temperature = temperature

    def generate_response(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        return response.choices[0].message.content
