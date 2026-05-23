import argparse
from typing import Tuple, Dict, Any

class CLIParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="PolyReasoner v4 CLI")
        self.subparsers = self.parser.add_subparsers(dest="command", help="Available commands")
        
        # /judge command
        self.judge_parser = self.subparsers.add_parser("judge", help="Run the Ensemble Security Judge")
        self.judge_parser.add_argument("--attack", type=str, required=True, help="The attack prompt")
        self.judge_parser.add_argument("--response", type=str, required=True, help="The target model's response")

    def parse(self) -> Tuple[str, Dict[str, Any]]:
        args = self.parser.parse_args()
        
        if args.command == "judge":
            return "judge", {"attack": args.attack, "response": args.response}
        
        return args.command, {}
