import os
import yaml
from pathlib import Path
from modes.judge_mode import JudgeMode

class PolyReasonerEnsemble:
    """
    The main library entry point for PolyReasoner.
    Allows declarative loading of model ensembles for autonomous research loops.
    """
    
    def __init__(self, profile: str = "security_judge"):
        self.profile_name = profile
        self.config = self._load_config(profile)
        
        # Instantiate modes
        self.judge_mode = JudgeMode(ensemble_config=self.config)
        
    def _load_config(self, profile: str) -> dict:
        config_path = Path(__file__).parent / "config" / "ensembles.yaml"
        if not config_path.exists():
            print(f"[!] Warning: Config file not found at {config_path}. Using default env vars.")
            return None
            
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            
        ensembles = data.get("ensembles", {})
        if profile not in ensembles:
            print(f"[!] Warning: Profile '{profile}' not found in ensembles.yaml. Using default env vars.")
            return None
            
        print(f"[*] PolyReasoner loaded ensemble profile: '{profile}'")
        return ensembles[profile]
        
    async def run_judge(self, attack_prompt: str, target_response: str) -> dict:
        """
        Runs the full Multi-Perspective Security Judge on a prompt/response pair.
        Returns a dict containing the JSON verdict.
        """
        return await self.judge_mode.run(attack_prompt, target_response)
