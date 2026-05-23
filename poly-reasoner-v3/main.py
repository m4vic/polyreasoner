import asyncio
import sys
from cli.parser import CLIParser
from cli.display import CLI
from modes.judge_mode import JudgeMode

async def main():
    parser = CLIParser()
    
    try:
        command, kwargs = parser.parse()
    except SystemExit:
        # Happens when user passes -h or invalid args
        return

    if command == "judge":
        judge_mode = JudgeMode()
        
        # Ensure backend is available
        if not judge_mode.backend.is_available():
            print("[!] Error: Backend is not reachable. Is Ollama running?")
            sys.exit(1)
            
        verdict = await judge_mode.run(**kwargs)
        CLI.print_json(verdict)
    else:
        print(f"Command '{command}' not yet implemented.")

if __name__ == "__main__":
    # Windows asyncio bug workaround
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    asyncio.run(main())
