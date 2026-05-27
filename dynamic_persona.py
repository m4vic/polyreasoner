import json
from prompts import AGENT_PROMPTS
from backend import BackendFactory

SYSTEM_PROMPT_GENERATOR = """You are a Persona Architect. Your job is to create a specialized agent system prompt for an evaluation ensemble.
Create a system prompt for the persona: '{persona}'.

The output MUST fit the following format exactly:
You are the {persona} analyst in a multi-perspective evaluation system.

[Include 3-4 specific bullet points detailing what this persona evaluates in the query]
[Specific instructions on their perspective, professional bias, and skepticisms]

Output your analysis as JSON:
{{
  "verdict": "select 3 relevant categories (e.g. viable/risky/weak or similar)",
  "confidence": 0.0 to 1.0,
  "key_findings": ["finding1", "finding2"],
  "concerns": ["concern1", "concern2"],
  "recommendation": "brief actionable advice"
}}

Provide ONLY the system prompt text itself. Do not wrap in markdown code blocks or JSON. Start directly with 'You are the...'
"""

async def generate_dynamic_prompt(persona: str, backend_type: str = None) -> str:
    """Generates a detailed agent system prompt for a custom persona name on the fly."""
    # Check if we already have it in static prompts
    persona_lower = persona.lower()
    if persona_lower in AGENT_PROMPTS:
        return AGENT_PROMPTS[persona_lower]

    print(f"[*] Dynamically drafting persona prompt for: '{persona}'...")
    
    try:
        # Use fast model to quickly design the persona prompt
        fast_backend = BackendFactory.create(backend_type=backend_type, tier="fast")
        
        prompt = SYSTEM_PROMPT_GENERATOR.format(persona=persona)
        generated_prompt = await fast_backend.complete(prompt, temperature=0.3)
        
        system_prompt = generated_prompt.strip()
        
        # Strip any markdown blocks if the LLM ignored instructions
        if system_prompt.startswith("```"):
            lines = system_prompt.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].startswith("```"):
                lines = lines[:-1]
            system_prompt = "\n".join(lines).strip()
            
        # Ensure it appends the target instruction line
        if not system_prompt.endswith("Idea to analyze:"):
            system_prompt += "\n\nIdea to analyze:"
            
        # Cache it dynamically to avoid re-generating
        AGENT_PROMPTS[persona_lower] = system_prompt
        return system_prompt
        
    except Exception as e:
        print(f"[!] Dynamic persona generation failed: {e}. Falling back to default template.")
        # Fallback template
        fallback = (
            f"You are the {persona} analyst in a multi-perspective evaluation system.\n\n"
            f"Analyze the idea strictly from the viewpoint, biases, professional expertise, and concerns of a {persona}.\n"
            f"Highlight specific challenges, opportunities, and standard practices associated with this domain.\n\n"
            f"Output your analysis as JSON:\n"
            f"{{\n"
            f"  \"verdict\": \"viable\" | \"risky\" | \"weak\",\n"
            f"  \"confidence\": 0.0 to 1.0,\n"
            f"  \"key_findings\": [\"finding1\", \"finding2\"],\n"
            f"  \"concerns\": [\"concern1\", \"concern2\"],\n"
            f"  \"recommendation\": \"actionable advice\"\n"
            f"}}\n\n"
            f"Idea to analyze:"
        )
        AGENT_PROMPTS[persona_lower] = fallback
        return fallback
