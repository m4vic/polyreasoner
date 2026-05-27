"""
Polyreasoner Agents
Individual perspective execution logic.
This module handles running specific analytical perspectives (like business, risk, or contrarian)
both synchronously (for local weights via llama-cpp) and asynchronously (for API / Ollama concurrency).
"""

import asyncio
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
# Import the dictionary containing system prompts for each agent perspective
from prompts import AGENT_PROMPTS


def run_agent(agent_name: str, idea: str, context: str, llm) -> dict:
    """
    Run a single agent synchronously and return its parsed analysis.
    Each agent is isolated - it sees only the core idea, custom prompt, and optional file context.
    """
    # 1. Validation: check if the requested agent exists in the preset prompts
    if agent_name not in AGENT_PROMPTS:
        return {
            "agent": agent_name,
            "error": f"Unknown agent: {agent_name}"
        }
    
    # 2. Build prompt: append the user's idea to the agent's specific instruction brief
    prompt = AGENT_PROMPTS[agent_name] + f"\n\n{idea}"
    
    # 3. Add context: if file/folder context is provided, append it to the prompt
    if context:
        prompt += f"\n\nAdditional context: {context}"
    
    try:
        # 4. Generate response using the synchronous LLM callable
        response = llm(
            prompt,
            max_tokens=200,      # Shorter limit keeps agent completions fast
            temperature=0.7,     # Add slight creativity balance
            stop=["</response>", "\n\n\n"] # Truncate early at common output bounds
        )
        
        # 5. Extract output string from llama-cpp model output dictionary
        output_text = response["choices"][0]["text"].strip()
        
        # 6. Parse JSON: attempt to extract a structured dict from the model's text
        result = parse_agent_output(output_text)
        # Store the agent name in the dict for identification in the ensemble
        result["agent"] = agent_name
        
        return result
        
    except Exception as e:
        # Gracefully capture exceptions so that a single agent crash doesn't halt the whole system
        return {
            "agent": agent_name,
            "error": str(e),
            "raw_output": output_text if 'output_text' in dir() else None
        }


def parse_agent_output(text: str) -> dict:
    """
    Extract JSON from agent response.
    Handles situations where models wrap JSON in markdown blocks (```json) or prefix it with conversational text.
    """
    # 1. Direct parsing: if the model returned clean JSON, load it immediately
    try:
        return json.loads(text)
    except:
        pass
    
    # 2. Regex extraction: search for curly braces matching a JSON object pattern
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except:
            pass
    
    # 3. Fallback: if JSON parsing failed, return raw output inside a generic dict
    return {
        "verdict": "unknown",
        "confidence": 0.5,
        "raw_output": text,
        "parse_failed": True
    }


def run_agents_sequential(agent_names: list, idea: str, context: str, llm) -> list:
    """
    Run multiple agents sequentially.
    Crucial for local llama-cpp backends since they cannot handle concurrent requests on the same GPU instance.
    """
    results = []
    
    # Loop through the list of requested agents and execute them one by one
    for name in agent_names:
        try:
            # Execute agent synchronously
            result = run_agent(name, idea, context, llm)
            results.append(result)
            print(f"  ✓ {name} complete")
        except Exception as e:
            # Record errors
            results.append({
                "agent": name,
                "error": str(e)
            })
            print(f"  ✗ {name} failed: {e}")
    
    return results


def format_agent_outputs(results: list) -> str:
    """
    Format list of agent JSON outputs into a clean Markdown block for the Synthesizer prompt.
    """
    formatted = []
    
    for result in results:
        # Identify which agent this result belongs to
        agent = result.get("agent", "unknown")
        
        # If an error occurred, append the error text
        if "error" in result:
            formatted.append(f"**{agent.upper()}**: Error - {result['error']}")
        else:
            # Format successful responses as syntax-highlighted markdown JSON blocks
            formatted.append(f"**{agent.upper()}**:\n```json\n{json.dumps(result, indent=2)}\n```")
    
    # Join all agent reports with double newlines
    return "\n\n".join(formatted)


async def run_agent_async(agent_name: str, query: str, context: str, backend) -> dict:
    """
    Run a single agent asynchronously and return its analysis.
    Essential for web APIs and Ollama, allowing concurrent inference.
    """
    # 1. Validation check / Dynamic persona generation
    if agent_name not in AGENT_PROMPTS:
        try:
            from dynamic_persona import generate_dynamic_prompt
            await generate_dynamic_prompt(agent_name)
        except Exception as e:
            return {
                "agent": agent_name,
                "error": f"Failed to generate dynamic prompt for {agent_name}: {e}"
            }
    
    # 2. Construct the agent's personalized prompt
    prompt = AGENT_PROMPTS[agent_name] + f"\n\n{query}"
    
    # 3. Add context
    if context:
        prompt += f"\n\nAdditional context:\n{context}"
        
    try:
        # 4. Invoke the async backend to generate the response text
        response_text = await backend.complete(
            prompt,
            temperature=0.7
        )
        
        # 5. Extract and parse the JSON response
        result = parse_agent_output(response_text.strip())
        result["agent"] = agent_name
        return result
        
    except Exception as e:
        # Handle exceptions gracefully to prevent full-loop failures
        return {
            "agent": agent_name,
            "error": str(e)
        }


async def run_agents_parallel(agent_names: list, query: str, context: str, backend) -> list:
    """
    Run all specified agents concurrently in parallel using asyncio.gather.
    Significantly reduces latency when using cloud APIs or high-performance local endpoints.
    """
    tasks = []
    # Create an async task for each agent in the list
    for name in agent_names:
        tasks.append(run_agent_async(name, query, context, backend))
        
    # Wait for all tasks to resolve concurrently and compile results
    results = await asyncio.gather(*tasks)
    return list(results)
