import json
import os
from urllib import request
import ollama
from typing import Dict, Any, Optional
from .base import LLMClient


class OllamaClient(LLMClient):
    def __init__(
        self, model: str = "llama2", host: str = None, temperature: float = 0.7
    ):
        self.model = model
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.temperature = temperature

    def generate_response(self, prompt: str) -> str:
        url = f"{self.host}/api/generate"
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        req = request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result["response"]
