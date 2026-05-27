import json
import re
from backend import BackendFactory

ROUTER_SYSTEM_PROMPT = """You are the Central Router for Polyreasoner, an advanced multi-agent system.
Your job is to analyze the user's input and determine which tool or mode is most appropriate.

Available Actions:
1. "chat": For greetings, small talk, general knowledge, simple explanations, or clarifying follow-ups.
2. "search": When the user asks about current events, recent news, real-time facts, or queries requiring external web lookup.
3. "analyze_folder": When the user asks to scan, review, explain, or analyze files in a folder, directory, or codebase.
4. "multiperspective": For complex decisions, business plans, or career advice that benefit from a structured multi-agent panel.
5. "manual": When the user explicitly requests specific custom perspectives (e.g. "analyze this from a developer and QA viewpoint").

Rules for Output:
You MUST output ONLY a JSON object matching this schema:
{
  "action": "chat" | "search" | "analyze_folder" | "multiperspective" | "manual",
  "reasoning": "brief explanation of routing choice",
  "query": "cleaned query or search terms",
  "preset": "career" | "business" | "decision", // only for "multiperspective"
  "agents": ["persona1", "persona2"],            // only for "manual"
  "folder_path": "extracted folder path or '.' as default" // only for "analyze_folder"
}

Do NOT wrap the output in markdown code blocks. Output raw JSON only.
"""

async def route_query(user_input: str, backend_type: str = None) -> dict:
    """Uses the fast model to route the user's natural query to the correct tool/pipeline."""
    # Fast path for very short messages/greetings to avoid LLM call
    lower_input = user_input.lower().strip()
    if lower_input in ["hi", "hello", "hey", "hola", "help", "exit", "quit", "status"]:
        return {
            "action": "chat",
            "reasoning": "Direct match for basic greeting/command.",
            "query": user_input
        }

    try:
        fast_backend = BackendFactory.create(backend_type=backend_type, tier="fast")
        
        prompt = f"{ROUTER_SYSTEM_PROMPT}\nUser Input: {user_input}\nJSON Decision:"
        response = await fast_backend.complete(prompt, temperature=0.1)
        
        # Clean JSON markdown blocks if any
        cleaned = response.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
            
        decision = json.loads(cleaned)
        return decision
        
    except Exception as e:
        # Fallback to chat mode on any error
        return {
            "action": "chat",
            "reasoning": f"Fallback due to router error: {e}",
            "query": user_input
        }
