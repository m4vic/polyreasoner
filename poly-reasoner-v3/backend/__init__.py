import os
from .base import BaseBackend
from .ollama_backend import OllamaBackend
from .api_backend import APIBackend

class BackendFactory:
    @staticmethod
    def create(backend_type: str = None, model: str = None) -> BaseBackend:
        # Default to env vars if not explicitly provided
        backend_type = backend_type or os.getenv("POLYREASONER_BACKEND", "ollama")
        
        if backend_type == "ollama":
            model = model or os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
            host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            return OllamaBackend(model_name=model, host=host)
        elif backend_type == "api":
            return APIBackend()
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")
