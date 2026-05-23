import json

class CLI:
    """Rich terminal UI stub for PolyReasoner."""
    
    @staticmethod
    def print_json(data: dict):
        print(json.dumps(data, indent=2))
