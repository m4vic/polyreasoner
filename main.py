import os
import sys
import asyncio
from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.styles import Style as PromptStyle

# Load configuration settings and API keys from the local .env file at startup
load_dotenv()

# Import the CLI argument parser
from cli.parser import CLIParser
# Import the terminal UI output handler
from cli.display import CLI
# Import the file and directory context parser
from cli.context_reader import read_directory_context
# Import the different modes Polyreasoner supports
from modes.judge_mode import JudgeMode
from modes.career_mode import CareerMode
from modes.decision_mode import DecisionMode
from modes.evaluate_mode import EvaluateMode
from modes.manual_mode import ManualMode
from modes.business_mode import BusinessMode

# Import settings manager to dynamically save and reload configurations
from config import SettingsManager, OLLAMA_HOST, OLLAMA_MODEL, LITELLM_MODEL, BACKEND_TYPE

class Polyreasoner:
    """Core multi-perspective reasoning engine wrapper for CLI and programmatic integration."""

    async def process_async(self, message: str) -> str:
        """Asynchronously processes a user message, routes it, and returns the result string."""
        active_settings = SettingsManager.load()
        backend = active_settings["POLYREASONER_BACKEND"]
        
        from router_agent import route_query
        routing_decision = await route_query(message, backend_type=backend)
        
        action = routing_decision.get("action", "chat")
        query = routing_decision.get("query", message)
        reasoning = routing_decision.get("reasoning", "")
        
        if action == "chat":
            from backend import BackendFactory
            backend_inst = BackendFactory.create(backend_type=backend, tier="fast")
            resp = await backend_inst.complete(query)
            return resp
            
        elif action == "search":
            from tools.web_search import web_search
            from backend import BackendFactory
            
            results = web_search(query, max_results=4)
            if results:
                context_str = "Web Search Results:\n"
                for idx, r in enumerate(results, 1):
                    context_str += f"[{idx}] {r['title']}\nURL: {r['url']}\nSnippet: {r['snippet']}\n\n"
            else:
                context_str = "No web search results found."
                
            smart_backend = BackendFactory.create(backend_type=backend, tier="smart")
            synthesis_prompt = (
                f"Synthesize a comprehensive, helpful response for the user query based on the following web search context. "
                f"Be factual, unbiased, and cite sources if necessary.\n\n"
                f"Context:\n{context_str}\n\n"
                f"User Query: {message}\n\n"
                f"Answer:"
            )
            resp = await smart_backend.complete(synthesis_prompt)
            return resp
            
        elif action == "analyze_folder":
            from cli.context_reader import read_directory_context
            from backend import BackendFactory
            
            folder_path = routing_decision.get("folder_path", ".")
            context_data = read_directory_context(folder_path)
            
            if context_data.startswith("[Error"):
                context_data = "Error reading context directory."
                
            smart_backend = BackendFactory.create(backend_type=backend, tier="smart")
            analysis_prompt = (
                f"Based on the following directory code context, answer the user's question or perform the requested analysis.\n\n"
                f"Context:\n{context_data}\n\n"
                f"User Query: {message}\n\n"
                f"Analysis:"
            )
            resp = await smart_backend.complete(analysis_prompt)
            return resp
            
        elif action == "multiperspective":
            preset = routing_decision.get("preset", "decision")
            model = active_settings["OLLAMA_SMART_MODEL"] if backend == "ollama" else active_settings["API_SMART_MODEL"]
            
            if preset == "career":
                mode = CareerMode(backend_override=backend, model_override=model)
            elif preset == "business":
                mode = BusinessMode(backend_override=backend, model_override=model)
            else:
                mode = DecisionMode(backend_override=backend, model_override=model)
                
            synthesis, agent_results = await mode.run(query=query)
            
            # Format agent results + synthesis for web view
            agent_details = "\n\n### Agent Perspectives Applied:\n"
            for res in agent_results:
                agent_name = res.get("agent", "specialist").capitalize()
                verdict = res.get("verdict", "N/A")
                findings = "\n".join([f"- {f}" for f in res.get("key_findings", [])])
                concerns = "\n".join([f"- {c}" for c in res.get("concerns", [])])
                agent_details += f"#### {agent_name} ({res.get('confidence', 1.0)*100:.0f}% Confidence)\n*Verdict: {verdict}*\n*Key Findings:*\n{findings}\n*Concerns:*\n{concerns}\n\n"
                
            return f"{synthesis}\n\n{agent_details}"
            
        elif action == "manual":
            agents = routing_decision.get("agents", ["business", "risk"])
            model = active_settings["OLLAMA_SMART_MODEL"] if backend == "ollama" else active_settings["API_SMART_MODEL"]
            
            mode = ManualMode(agent_names=agents, backend_override=backend, model_override=model)
            synthesis, agent_results = await mode.run(query=query)
            
            agent_details = "\n\n### Custom Agent Perspectives:\n"
            for res in agent_results:
                agent_name = res.get("agent", "specialist").capitalize()
                verdict = res.get("verdict", "N/A")
                findings = "\n".join([f"- {f}" for f in res.get("key_findings", [])])
                concerns = "\n".join([f"- {c}" for c in res.get("concerns", [])])
                agent_details += f"#### {agent_name} ({res.get('confidence', 1.0)*100:.0f}% Confidence)\n*Verdict: {verdict}*\n*Key Findings:*\n{findings}\n*Concerns:*\n{concerns}\n\n"
                
            return f"{synthesis}\n\n{agent_details}"
            
        return "Unknown action routed."
        
    def process(self, message: str) -> str:
        """Synchronous wrapper to run the async process pipeline, handling running event loops."""
        import threading
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        if loop.is_running():
            result = []
            def run():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                res = new_loop.run_until_complete(self.process_async(message))
                result.append(res)
                new_loop.close()
            t = threading.Thread(target=run)
            t.start()
            t.join()
            return result[0]
        else:
            return loop.run_until_complete(self.process_async(message))

async def run_preset_analysis(command: str, query: str, context: str = None, backend_override: str = None, model_override: str = None, agents_override: list = None):
    """Orchestrates running a selected reasoning mode preset and displaying results."""
    # Instantiate the corresponding mode class based on the parsed command
    if command == "career":
        mode = CareerMode(backend_override=backend_override, model_override=model_override)
    elif command == "business":
        mode = BusinessMode(backend_override=backend_override, model_override=model_override)
    elif command == "decision":
        mode = DecisionMode(backend_override=backend_override, model_override=model_override)
    elif command == "idea_evaluation":
        mode = EvaluateMode(backend_override=backend_override, model_override=model_override)
    elif command == "manual":
        mode = ManualMode(agent_names=agents_override or ["business", "risk"], backend_override=backend_override, model_override=model_override)
    else:
        print(f"[!] Error: Command '{command}' not recognized.")
        return

    # Check backend availability before executing models
    if not mode.backend.is_available():
        print(f"[!] Error: Backend '{backend_override or os.getenv('POLYREASONER_BACKEND', 'ollama')}' is not reachable.")
        print("    Please check if Ollama is running locally, or if your API keys are configured.")
        return
        
    try:
        # Run the parallel agent pipeline and synthesize results
        synthesis, agent_results = await mode.run(query=query, context=context)
        
        # Render status grid and final synthesis markdown report
        CLI.print_agent_status(agent_results)
        CLI.print_synthesis(synthesis, query)
    except Exception as e:
        print(f"[!] Error during run execution: {e}")
        import traceback
        traceback.print_exc()

def interactive_settings():
    """Interactively guides the user to modify and save persistent configuration settings."""
    settings = SettingsManager.load()
    
    print("\n" + "="*40)
    print("PolyReasoner v4 Settings Editor")
    print("="*40)
    
    # 1. Backend Type
    backend = input(f"Default Backend [current: {settings.get('POLYREASONER_BACKEND')} (ollama/api)]: ").strip().lower()
    if backend in ["ollama", "api"]:
        settings["POLYREASONER_BACKEND"] = backend
    
    # 2. Ollama settings
    ollama_host = input(f"Ollama Host URL [current: {settings.get('OLLAMA_HOST')}]: ").strip()
    if ollama_host:
        settings["OLLAMA_HOST"] = ollama_host
        
    ollama_model = input(f"Ollama Model (Default) [current: {settings.get('OLLAMA_MODEL')}]: ").strip()
    if ollama_model:
        settings["OLLAMA_MODEL"] = ollama_model
        
    ollama_fast = input(f"Ollama Model (Fast/Weak) [current: {settings.get('OLLAMA_FAST_MODEL')}]: ").strip()
    if ollama_fast:
        settings["OLLAMA_FAST_MODEL"] = ollama_fast
        
    ollama_smart = input(f"Ollama Model (Smart/Heavy) [current: {settings.get('OLLAMA_SMART_MODEL')}]: ").strip()
    if ollama_smart:
        settings["OLLAMA_SMART_MODEL"] = ollama_smart
        
    # 3. LiteLLM cloud settings
    litellm_model = input(f"LiteLLM Model (Default) [current: {settings.get('LITELLM_MODEL')}]: ").strip()
    if litellm_model:
        settings["LITELLM_MODEL"] = litellm_model
        
    litellm_fast = input(f"LiteLLM Model (Fast/Weak) [current: {settings.get('API_FAST_MODEL')}]: ").strip()
    if litellm_fast:
        settings["API_FAST_MODEL"] = litellm_fast
        
    litellm_smart = input(f"LiteLLM Model (Smart/Heavy) [current: {settings.get('API_SMART_MODEL')}]: ").strip()
    if litellm_smart:
        settings["API_SMART_MODEL"] = litellm_smart
        
    # 4. API keys
    print("\nAPI Keys Setup (Press enter to keep current value):")
    api_keys = settings.get("API_KEYS", {})
    
    openai_key = input("OpenAI API Key: ").strip()
    if openai_key:
        api_keys["OPENAI_API_KEY"] = openai_key
        
    anthropic_key = input("Anthropic API Key: ").strip()
    if anthropic_key:
        api_keys["ANTHROPIC_API_KEY"] = anthropic_key
        
    gemini_key = input("Gemini API Key: ").strip()
    if gemini_key:
        api_keys["GEMINI_API_KEY"] = gemini_key
        
    settings["API_KEYS"] = api_keys
    
    # Save to file
    if SettingsManager.save(settings):
        print("\n[OK] Settings persistently saved to settings.json!")
        # Dynamically reapply to current environment
        os.environ["POLYREASONER_BACKEND"] = settings["POLYREASONER_BACKEND"]
        os.environ["OLLAMA_HOST"] = settings["OLLAMA_HOST"]
        os.environ["OLLAMA_MODEL"] = settings["OLLAMA_MODEL"]
        os.environ["OLLAMA_FAST_MODEL"] = settings.get("OLLAMA_FAST_MODEL", "llama3.1:8b")
        os.environ["OLLAMA_SMART_MODEL"] = settings.get("OLLAMA_SMART_MODEL", "qwen2.5-coder:14b")
        os.environ["LITELLM_MODEL"] = settings["LITELLM_MODEL"]
        os.environ["API_FAST_MODEL"] = settings.get("API_FAST_MODEL", "gpt-4o-mini")
        os.environ["API_SMART_MODEL"] = settings.get("API_SMART_MODEL", "gpt-4o")
        for k, v in api_keys.items():
            if v:
                os.environ[k] = v
    else:
        print("\n[!] Failed to save settings.")
    print("="*40 + "\n")

def is_casual_query(query: str) -> bool:
    """Checks if a user query is just a simple greeting or casual chat."""
    clean = query.strip().lower().rstrip("?").rstrip("!").rstrip(".")
    greetings = {
        "hi", "hello", "hey", "yo", "sup", "greetings", "howdy", "hola", 
        "namaste", "test", "ping", "hello world", "good morning", 
        "good afternoon", "good evening", "welcome"
    }
    if clean in greetings:
        return True
    words = clean.split()
    if len(words) <= 2 and (clean in {"what's up", "whats up", "how's it going", "hows it going"} or len(clean) < 4):
        return True
    return False

def extract_agents_override(query: str) -> tuple[str, list[str] | None]:
    """Helper to parse a dynamic '--agents agent1,agent2' flag from user input."""
    if "--agents" in query:
        parts = query.split("--agents")
        query_text = parts[0].strip()
        agents_str = parts[1].strip()
        if " " in agents_str:
            agents_str = agents_str.split()[0]
        agents_list = [a.strip().lower() for a in agents_str.split(",") if a.strip()]
        return query_text, agents_list
    return query, None

async def interactive_shell():
    """Launches the interactive CLI REPL shell for persistent, multi-turn reasoning."""
    CLI.print_banner()
    
    # Print current active settings on startup
    active_settings = SettingsManager.load()
    CLI.print_settings(active_settings)
    
    # Set up prompt autocomplete session
    completer = WordCompleter([
        "/career", "/business", "/decision", "/manual", "/settings", "/help", "/exit", "/quit"
    ], ignore_case=True)
    
    session = PromptSession(completer=completer)
    style = PromptStyle.from_dict({
        'prompt': 'ansibrightmagenta bold',
    })
    
    print("[*] Ready for multi-perspective queries. Enter a question or /help.")
    
    while True:
        try:
            # Multi-turn prompt input
            user_input = await session.prompt_async("poly> ", style=style)
            user_input = user_input.strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ["/exit", "/quit"]:
                print("Goodbye!")
                break
                
            if user_input.lower() == "/help":
                print("\nPolyReasoner Shell Commands:")
                print("  /career <query>   - Run career trajectory and risk analysis")
                print("  /business <query> - Analyze startup plans, ethics, risk and viability")
                print("  /decision <query> - Standard decision trade-off analysis")
                print("  /manual <query> --agents <list> - Run custom dynamic agent list")
                print("  /model <name>     - Quick-switch the active Ollama model")
                print("  /status           - Show current active settings")
                print("  /settings         - Open persistent settings editor")
                print("  /exit, /quit      - Exit the shell\n")
                continue
            
            # /status: quick inline settings view
            if user_input.lower() == "/status":
                CLI.print_settings(SettingsManager.load())
                continue
            
            # /model: quick-switch the active model without full settings wizard
            if user_input.lower().startswith("/model"):
                parts = user_input.split(maxsplit=1)
                if len(parts) < 2 or not parts[1].strip():
                    current = SettingsManager.load()
                    print(f"[*] Current model: {current.get('OLLAMA_MODEL', 'N/A')}")
                    print("    Usage: /model <name>  (e.g. /model deepseek-r1:8b)")
                else:
                    new_model = parts[1].strip()
                    current = SettingsManager.load()
                    current["OLLAMA_MODEL"] = new_model
                    SettingsManager.save(current)
                    os.environ["OLLAMA_MODEL"] = new_model
                    print(f"[OK] Active model switched to: {new_model}")
                continue
                
            if user_input.lower() == "/settings":
                interactive_settings()
                continue
                
            # Parse slash commands dynamically inside the REPL loop
            if user_input.startswith("/"):
                space_index = user_input.find(" ")
                cmd = user_input[1:space_index] if space_index != -1 else user_input[1:]
                query = user_input[space_index+1:].strip() if space_index != -1 else ""
                
                # Check for manual command constraints
                if cmd == "manual":
                    query, agents_override = extract_agents_override(query)
                    if not agents_override:
                        print("[!] /manual command requires specifying active agents using --agents (e.g. /manual What is AI? --agents risk,business)")
                        continue
                    if not query:
                        query = input("Enter your question: ").strip()
                        if not query:
                            print("[!] Query cannot be empty.")
                            continue
                else:
                    query, agents_override = extract_agents_override(query)
                
                if cmd in ["career", "business", "decision", "manual"]:
                    # Load active settings
                    active_settings = SettingsManager.load()
                    backend = active_settings["POLYREASONER_BACKEND"]
                    model = active_settings["OLLAMA_MODEL"] if backend == "ollama" else active_settings["LITELLM_MODEL"]
                    
                    # Handle casual query routing gateway
                    if query and is_casual_query(query):
                        from backend import BackendFactory
                        backend_inst = BackendFactory.create(backend_type=backend, model=model)
                        print(f"[*] Fast-routing casual chat using {backend.upper()} ({model})...")
                        resp = await backend_inst.complete(query)
                        print(f"\nResponse: {resp}\n")
                        continue
                        
                    if not query and cmd != "manual":
                        query = input("Enter your question: ").strip()
                        if not query:
                            print("[!] Query cannot be empty.")
                            continue
                    
                    print(f"[*] Running /{cmd} analysis using {backend.upper()} ({model})...")
                    run_cmd = cmd
                    if agents_override and cmd != "manual":
                        run_cmd = "manual"
                        
                    await run_preset_analysis(run_cmd, query, backend_override=backend, model_override=model, agents_override=agents_override)
                else:
                    print(f"[!] Command '/{cmd}' not recognized. Type /help for details.")
            else:
                # Default behavior: run decision mode if no command prefix is provided
                active_settings = SettingsManager.load()
                backend = active_settings["POLYREASONER_BACKEND"]
                
                print("[*] Routing query autonomously...")
                from router_agent import route_query
                routing_decision = await route_query(user_input, backend_type=backend)
                
                action = routing_decision.get("action", "chat")
                query = routing_decision.get("query", user_input)
                reasoning = routing_decision.get("reasoning", "")
                
                print(f"[*] Router Selected: {action.upper()} | Reasoning: {reasoning}")
                
                if action == "chat":
                    from backend import BackendFactory
                    backend_inst = BackendFactory.create(backend_type=backend, tier="fast")
                    print(f"[*] Fast-routing chat response using {backend.upper()} ({backend_inst.model_name})...")
                    resp = await backend_inst.complete(query)
                    print(f"\nResponse: {resp}\n")
                    
                elif action == "search":
                    from tools.web_search import web_search
                    from backend import BackendFactory
                    
                    print(f"[*] Searching the web for: '{query}'...")
                    results = web_search(query, max_results=4)
                    
                    if results:
                        context_str = "Web Search Results:\n"
                        for idx, r in enumerate(results, 1):
                            context_str += f"[{idx}] {r['title']}\nURL: {r['url']}\nSnippet: {r['snippet']}\n\n"
                    else:
                        context_str = "No web search results found."
                        
                    smart_backend = BackendFactory.create(backend_type=backend, tier="smart")
                    print(f"[*] Synthesizing answer using smart model ({smart_backend.model_name})...")
                    
                    synthesis_prompt = (
                        f"Synthesize a comprehensive, helpful response for the user query based on the following web search context. "
                        f"Be factual, unbiased, and cite sources if necessary.\n\n"
                        f"Context:\n{context_str}\n\n"
                        f"User Query: {user_input}\n\n"
                        f"Answer:"
                    )
                    resp = await smart_backend.complete(synthesis_prompt)
                    print(f"\nWeb-Informed Response:\n{resp}\n")
                    
                elif action == "analyze_folder":
                    from cli.context_reader import read_directory_context
                    from backend import BackendFactory
                    
                    folder_path = routing_decision.get("folder_path", ".")
                    print(f"[*] Scanning context from folder: '{folder_path}'...")
                    context_data = read_directory_context(folder_path)
                    
                    if context_data.startswith("[Error"):
                        print(f"[!] Warning: Folder read error: {context_data}")
                        context_data = None
                    else:
                        print(f"[*] Ingested {len(context_data)} characters of folder context.")
                        
                    smart_backend = BackendFactory.create(backend_type=backend, tier="smart")
                    print(f"[*] Analyzing folder context using smart model ({smart_backend.model_name})...")
                    
                    analysis_prompt = (
                        f"Based on the following directory code context, answer the user's question or perform the requested analysis.\n\n"
                        f"Context:\n{context_data}\n\n"
                        f"User Query: {user_input}\n\n"
                        f"Analysis:"
                    )
                    resp = await smart_backend.complete(analysis_prompt)
                    print(f"\nCode Analysis Response:\n{resp}\n")
                    
                elif action == "multiperspective":
                    preset = routing_decision.get("preset", "decision")
                    model = active_settings["OLLAMA_SMART_MODEL"] if backend == "ollama" else active_settings["API_SMART_MODEL"]
                    print(f"[*] Launching preset /{preset} analysis using {backend.upper()} ({model})...")
                    await run_preset_analysis(preset, query, backend_override=backend, model_override=model)
                    
                elif action == "manual":
                    agents = routing_decision.get("agents", ["business", "risk"])
                    model = active_settings["OLLAMA_SMART_MODEL"] if backend == "ollama" else active_settings["API_SMART_MODEL"]
                    print(f"[*] Launching dynamic custom agents panel: {', '.join(agents)}")
                    await run_preset_analysis("manual", query, backend_override=backend, model_override=model, agents_override=agents)
                
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"[!] Shell error: {e}")

async def main():
    """
    Main asynchronous CLI driver. Parses inputs, loads file context,
    sets up API configurations, executes the reasoning presets, and prints output.
    """
    # Create parser instance
    parser = CLIParser()
    
    try:
        # Parse command line arguments. Handles slash command conversions (/career -> career)
        command, kwargs = parser.parse()
    except SystemExit:
        # Graceful exit on parser helper prints
        return

    # -------------------------------------------------------------
    # Execution Path 1: Interactive REPL Shell Mode
    # -------------------------------------------------------------
    if command is None:
        await interactive_shell()
        return

    # -------------------------------------------------------------
    # Execution Path 2: Persistent Settings Configuration Editor
    # -------------------------------------------------------------
    if command == "settings":
        interactive_settings()
        return

    # -------------------------------------------------------------
    # Execution Path 3: Ensemble Security Judge (ASRT compatibility)
    # -------------------------------------------------------------
    if command == "judge":
        # Instantiate the safety/jailbreak evaluation judge
        judge_mode = JudgeMode()
        
        # Verify the model backend is active and reachable
        if not judge_mode.safety_backend.is_available():
            print("[!] Error: Backend is not reachable. Is Ollama running?")
            sys.exit(1)
            
        # Extract evaluation arguments
        attack = kwargs.get("attack")
        response = kwargs.get("response")
        
        # Run judge analysis concurrently and synthesize the verdict
        verdict = await judge_mode.run(attack=attack, response=response)
        
        # Print the final formatted JSON verdict output to the terminal
        CLI.print_json(verdict)
        
    # -------------------------------------------------------------
    # Execution Path 4: Direct Single-Shot Command Presets
    # -------------------------------------------------------------
    elif command in ["career", "business", "decision", "idea_evaluation", "manual"]:
        # Dynamic configuration overrides via arguments
        if kwargs.get("api_key"):
            key = kwargs["api_key"]
            os.environ["OPENAI_API_KEY"] = key
            os.environ["ANTHROPIC_API_KEY"] = key
            os.environ["GEMINI_API_KEY"] = key
            
        if kwargs.get("ollama_host"):
            os.environ["OLLAMA_HOST"] = kwargs["ollama_host"]
            
        # Context Ingestion
        context = None
        if kwargs.get("dir"):
            dir_path = kwargs["dir"]
            print(f"[*] Reading context directory/file: {dir_path}...")
            context = read_directory_context(dir_path)
            if context.startswith("[Error"):
                print(f"[!] Warning: Context read error: {context}")
                context = None
            else:
                print(f"[*] Ingested {len(context)} chars of context.")

        # Backend and Model Overrides
        backend_override = kwargs.get("backend")
        model_override = kwargs.get("model")
        agents_list = None
        
        if command == "manual":
            agents_list = [a.strip() for a in kwargs["agents"].split(",")]
            
        query = kwargs.get("query")
        if not query:
            # If subcommand is run without a query argument, start shell or prompt
            print(f"[*] Running interactive prompt for command: /{command}")
            query = input("Enter your question: ").strip()
            if not query:
                print("[!] Query cannot be empty.")
                return

        await run_preset_analysis(
            command=command,
            query=query,
            context=context,
            backend_override=backend_override,
            model_override=model_override,
            agents_override=agents_list
        )

def cli_entry():
    """Global script entry point registered in pyproject.toml."""
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())

if __name__ == "__main__":
    cli_entry()



