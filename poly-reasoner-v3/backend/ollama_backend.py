import json
from typing import AsyncIterator, Dict, Any
import ollama
from .base import BaseBackend

class OllamaBackend(BaseBackend):
    def __init__(self, model_name: str = "qwen2.5-coder:7b", host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.client = ollama.AsyncClient(host=host)
        
    async def complete(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = await self.client.chat(
            model=self.model_name,
            messages=messages,
            options=kwargs
        )
        return response['message']['content']

    async def complete_json(self, prompt: str, system_prompt: str = None, **kwargs) -> Dict[str, Any]:
        """
        Forces Ollama to output JSON via the format parameter.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = await self.client.chat(
            model=self.model_name,
            messages=messages,
            format='json',
            options=kwargs
        )
        
        try:
            return json.loads(response['message']['content'])
        except json.JSONDecodeError:
            # Fallback if the model hallucinated outside JSON bounds
            return {"error": "Failed to parse JSON", "raw": response['message']['content']}

    async def stream(self, prompt: str, system_prompt: str = None, **kwargs) -> AsyncIterator[str]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        async for chunk in await self.client.chat(
            model=self.model_name,
            messages=messages,
            stream=True,
            options=kwargs
        ):
            yield chunk['message']['content']

    def is_available(self) -> bool:
        try:
            # Quick sync check to see if daemon is responding
            import requests
            resp = requests.get(self.client.host)
            return resp.status_code == 200
        except Exception:
            return False
