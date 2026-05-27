import asyncio
from typing import List, Dict, Any, Tuple
# Import the parallel agent execution helper and output formatter
from agents import run_agents_parallel, format_agent_outputs
# Import synthesis prompt which directs the LLM on how to combine perspectives
from prompts import SYNTHESIS_PROMPT
# Import backend factory to instantiate the required LLM backend
from backend import BackendFactory

class ReasoningMode:
    """
    Core engine for all decision-making modes.
    Manages loading a backend, executing agents asynchronously, and synthesizing results.
    """
    def __init__(self, agent_names: List[str], backend_override: str = None, model_override: str = None):
        # Store list of active agents/perspectives for this run
        self.agent_names = agent_names
        
        # Instantiate LLM backend. If overrides are None, falls back to env vars (.env)
        self.backend = BackendFactory.create(backend_type=backend_override, model=model_override)
        
    async def run(self, query: str, context: str = None) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Executes the multi-agent pipeline asynchronously.
        1. Runs each active agent concurrently (via run_agents_parallel).
        2. Formats agent outputs.
        3. Calls synthesizer LLM to produce a final report showing agreements, conflicts, and trade-offs.
        """
        # Print informational logs to terminal
        print(f"[*] Activating perspectives: {', '.join(self.agent_names)}")
        print("[*] Running agents in parallel...")
        
        # Run all specified agents in parallel using the async backend
        agent_results = await run_agents_parallel(
            agent_names=self.agent_names,
            query=query,
            context=context,
            backend=self.backend
        )
        
        # Check individual agent results for errors and print statuses
        for res in agent_results:
            if "error" in res:
                print(f"    [!] Warning: Agent '{res['agent']}' failed: {res['error']}")
            else:
                print(f"    ✓ Perspective '{res['agent']}' analysis complete")
                
        # Format the JSON outputs from all successful agents into a single string block
        formatted_outputs = format_agent_outputs(agent_results)
        
        # Construct the synthesis prompt using the formatted agent outputs and original query
        synth_prompt = SYNTHESIS_PROMPT.format(
            agent_outputs=formatted_outputs,
            original_query=query
        )
        
        print("[*] Synthesizing perspectives into final trade-offs...")
        
        # Pass synthesis prompt to the LLM backend to construct final markdown evaluation
        synthesis = await self.backend.complete(synth_prompt)
        
        # Return both the synthesis markdown and raw agent results (useful for web UI/pipelines)
        return synthesis, agent_results
