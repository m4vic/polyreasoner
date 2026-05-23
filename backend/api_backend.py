from typing import AsyncIterator, Dict, Any
from .base import BaseBackend

class APIBackend(BaseBackend):
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        
    async def complete(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """LiteLLM integration will go here."""
        raise NotImplementedError("API Backend not yet implemented. Use OllamaBackend.")

    async def complete_json(self, prompt: str, system_prompt: str = None, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("API Backend not yet implemented.")

    async def stream(self, prompt: str, system_prompt: str = None, **kwargs) -> AsyncIterator[str]:
        raise NotImplementedError("API Backend not yet implemented.")
        yield ""

    def is_available(self) -> bool:
        return False
