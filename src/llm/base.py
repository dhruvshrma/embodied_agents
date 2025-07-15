from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    def generate_response(self, prompt: str) -> str:
        """Generates a response from the LLM based on a single prompt."""
        ...
