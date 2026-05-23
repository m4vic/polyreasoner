from abc import ABC, abstractmethod
from typing import AsyncIterator, Dict, Any

class BaseBackend(ABC):
    """
    Abstract base class for all PolyReasoner backends.
    """
    
    @abstractmethod
    async def complete(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """
        Standard async text completion.
        """
        pass

    @abstractmethod
    async def complete_json(self, prompt: str, system_prompt: str = None, **kwargs) -> Dict[str, Any]:
        """
        Forces the model to output strict JSON.
        """
        pass

    @abstractmethod
    async def stream(self, prompt: str, system_prompt: str = None, **kwargs) -> AsyncIterator[str]:
        """
        Stream output tokens.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Checks if the backend is reachable (e.g. Ollama daemon running, API key valid).
        """
        pass
