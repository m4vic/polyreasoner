import os
from .base import BaseBackend
from .ollama_backend import OllamaBackend
from .api_backend import APIBackend

class BackendFactory:
    @staticmethod
    def create(backend_type: str = None, model: str = None, tier: str = None) -> BaseBackend:
        from config import SettingsManager
        settings = SettingsManager.load()
        
        # Default to env vars or settings if not explicitly provided
        backend_type = backend_type or settings.get("POLYREASONER_BACKEND", "ollama")
        
        if backend_type == "ollama":
            if not model:
                if tier == "fast":
                    model = settings.get("OLLAMA_FAST_MODEL", "llama3.1:8b")
                elif tier == "smart":
                    model = settings.get("OLLAMA_SMART_MODEL", "qwen2.5-coder:14b")
                else:
                    model = settings.get("OLLAMA_MODEL", "qwen2.5-coder:14b")
            host = settings.get("OLLAMA_HOST", "http://localhost:11434")
            return OllamaBackend(model_name=model, host=host)
            
        elif backend_type == "api":
            if not model:
                if tier == "fast":
                    model = settings.get("API_FAST_MODEL", "gpt-4o-mini")
                elif tier == "smart":
                    model = settings.get("API_SMART_MODEL", "gpt-4o")
                else:
                    model = settings.get("LITELLM_MODEL", "gpt-4o")
            return APIBackend(model_name=model)
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")
