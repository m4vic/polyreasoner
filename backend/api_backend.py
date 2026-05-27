import json
import litellm
from typing import AsyncIterator, Dict, Any
from .base import BaseBackend

class APIBackend(BaseBackend):
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        
    async def complete(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = await litellm.acompletion(
            model=self.model_name,
            messages=messages,
            **kwargs
        )
        return response.choices[0].message.content or ""

    async def complete_json(self, prompt: str, system_prompt: str = None, **kwargs) -> Dict[str, Any]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = await litellm.acompletion(
            model=self.model_name,
            messages=messages,
            response_format={"type": "json_object"},
            **kwargs
        )
        content = response.choices[0].message.content or ""
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"error": "Failed to parse JSON", "raw": content}

    async def stream(self, prompt: str, system_prompt: str = None, **kwargs) -> AsyncIterator[str]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = await litellm.acompletion(
            model=self.model_name,
            messages=messages,
            stream=True,
            **kwargs
        )
        async for chunk in response:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield delta

    def is_available(self) -> bool:
        return True

