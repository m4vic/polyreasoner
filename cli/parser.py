import argparse
import sys
from typing import Tuple, Dict, Any

class CLIParser:
    def __init__(self):
        # Initialize the argument parser with a description of the CLI tool
        self.parser = argparse.ArgumentParser(description="PolyReasoner v4 CLI - Multi-Perspective Decision Engine")
        
        # Subparsers allow us to have commands like 'poly career' or 'poly decision'
        # dest="command" stores which subcommand was chosen in the args object
        self.subparsers = self.parser.add_subparsers(dest="command", help="Available commands")
        
        # Helper function to add common arguments to subcommands to avoid duplicate code
        def add_common_args(subparser):
            # Positional argument: the core question or idea the user wants to evaluate (optional for shell)
            subparser.add_argument("query", type=str, nargs="?", default=None, help="The query or decision question")
            
            # Optional directory or file path to pull context into the prompt automatically
            subparser.add_argument("--dir", type=str, default=None, help="Path to file or folder for context ingestion")
            
            # Optional LLM model override (e.g. gpt-4o, claude-3-5-sonnet, qwen2.5-coder:14b)
            subparser.add_argument("--model", type=str, default=None, help="Override LLM model name")
            
            # Optional backend override: 'ollama' for local, 'api' for LiteLLM
            subparser.add_argument("--backend", type=str, choices=["ollama", "api"], default=None, help="Override backend (ollama or api)")
            
            # Optional API key input to dynamically configure API clients
            subparser.add_argument("--api-key", type=str, default=None, help="API key for Litellm (OpenAI/Claude)")
            
            # Optional Ollama server URL host override
            subparser.add_argument("--ollama-host", type=str, default=None, help="Host address for Ollama")

        # -------------------------------------------------------------
        # /career command: evaluate career paths, choices, or jobs
        # -------------------------------------------------------------
        career_parser = self.subparsers.add_parser("career", help="Evaluate career decisions and options")
        add_common_args(career_parser)
        
        # -------------------------------------------------------------
        # /business command: analyze business or project ideas
        # -------------------------------------------------------------
        business_parser = self.subparsers.add_parser("business", help="Evaluate business plans, startups, or ideas")
        add_common_args(business_parser)
        
        # -------------------------------------------------------------
        # /decision command: generic multi-perspective decision analysis
        # -------------------------------------------------------------
        decision_parser = self.subparsers.add_parser("decision", help="Standard multi-perspective decision analysis")
        add_common_args(decision_parser)
        
        # -------------------------------------------------------------
        # /idea_evaluation command: analyze business or project ideas (legacy)
        # -------------------------------------------------------------
        idea_parser = self.subparsers.add_parser("idea_evaluation", help="Evaluate business ideas or projects (legacy)")
        add_common_args(idea_parser)
        
        # -------------------------------------------------------------
        # /manual command: evaluate with manually specified agent lenses
        # -------------------------------------------------------------
        manual_parser = self.subparsers.add_parser("manual", help="Run with manually selected agents")
        add_common_args(manual_parser)
        # Unique arg for manual mode: comma-separated string list of agents (e.g. business,risk)
        manual_parser.add_argument("--agents", type=str, required=True, help="Comma-separated list of agents to run (e.g. risk,business,feasibility)")
        
        # -------------------------------------------------------------
        # /settings command: edit persistent configuration settings
        # -------------------------------------------------------------
        settings_parser = self.subparsers.add_parser("settings", help="Modify persistent backend, models, and API key settings")

        # -------------------------------------------------------------
        # /judge command: preserves compatibility with ASRT red-teaming pipeline
        # -------------------------------------------------------------
        judge_parser = self.subparsers.add_parser("judge", help="Run the Ensemble Security Judge (ASRT compatibility)")
        # The judge mode evaluates a adversarial attack and model response
        judge_parser.add_argument("--attack", type=str, required=True, help="The attack prompt")
        judge_parser.add_argument("--response", type=str, required=True, help="The target model's response")

    def parse(self, args_list=None) -> Tuple[str, Dict[str, Any]]:
        """
        Parses arguments, pre-processing slash prefixes like /career to career.
        This enables users to type slash commands in the CLI identically to slack/discord.
        """
        # If no explicit args_list is passed, read from sys.argv (command line arguments)
        if args_list is None:
            args_list = sys.argv[1:]
            
        processed_args = []
        # Loop through command-line tokens to clean slash commands
        for arg in args_list:
            # If the argument starts with '/' (e.g. /career) and isn't a Windows path
            if arg.startswith("/") and not arg.startswith("//") and len(arg) > 1 and not (arg.startswith("/dev") or arg.startswith("/mnt") or ":" in arg):
                # Remove the leading slash
                cmd_candidate = arg[1:]
                # Check if it matches one of our valid subcommands
                if cmd_candidate in ["career", "business", "decision", "idea_evaluation", "manual", "judge", "settings"]:
                    processed_args.append(cmd_candidate)
                    continue
            processed_args.append(arg)
            
        # If no arguments or command was provided, return None to trigger interactive shell mode
        if not processed_args:
            return None, {}
            
        # Parse arguments with argparse
        parsed_args = self.parser.parse_args(processed_args)
        
        # Extract the subcommand name (career, decision, etc.)
        cmd = parsed_args.command
        # Convert Namespace object to a standard Python dictionary
        kwargs = vars(parsed_args)
        # Remove the subcommand name from the dict since it is returned separately
        kwargs.pop("command")
        
        # Return subcommand name and its key-value arguments
        return cmd, kwargs
