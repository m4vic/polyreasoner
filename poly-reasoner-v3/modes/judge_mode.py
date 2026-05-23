import asyncio
import json
from backend import BackendFactory
from backend.specialist_moe import SpecialistMoE
from judge_prompts import SAFETY_JUDGE_PROMPT, JAILBREAK_JUDGE_PROMPT, SYNTHESIZER_JUDGE_PROMPT

class JudgeMode:
    def __init__(self, ensemble_config: dict = None):
        """
        Initializes the Judge. If ensemble_config is provided, it uses those 
        specific models for each judge role. Otherwise it defaults to env vars.
        """
        if ensemble_config:
            self.safety_backend = BackendFactory.create(
                backend_type=ensemble_config.get("safety_judge", {}).get("backend"),
                model=ensemble_config.get("safety_judge", {}).get("model")
            )
            self.jailbreak_backend = BackendFactory.create(
                backend_type=ensemble_config.get("jailbreak_judge", {}).get("backend"),
                model=ensemble_config.get("jailbreak_judge", {}).get("model")
            )
            self.synth_backend = BackendFactory.create(
                backend_type=ensemble_config.get("synthesizer", {}).get("backend"),
                model=ensemble_config.get("synthesizer", {}).get("model")
            )
        else:
            default_backend = BackendFactory.create()
            self.safety_backend = default_backend
            self.jailbreak_backend = default_backend
            self.synth_backend = default_backend
            
        # New: 5-dimensional BERT specialist MoE (replaces old 6-binary ExpertEnsemble)
        self.moe = SpecialistMoE()
        
    async def _run_safety_judge(self, attack: str, response: str) -> str:
        prompt = f"Attack Prompt:\n{attack}\n\nTarget Response:\n{response}"
        return await self.safety_backend.complete(prompt, system_prompt=SAFETY_JUDGE_PROMPT)

    async def _run_jailbreak_judge(self, attack: str, response: str) -> str:
        prompt = f"Attack Prompt:\n{attack}\n\nTarget Response:\n{response}"
        return await self.jailbreak_backend.complete(prompt, system_prompt=JAILBREAK_JUDGE_PROMPT)

    async def run(self, attack: str, response: str) -> dict:
        print("[*] Running Ensemble Security Judge...")
        
        # Run specialized LLM judges concurrently
        safety_task = asyncio.create_task(self._run_safety_judge(attack, response))
        jailbreak_task = asyncio.create_task(self._run_jailbreak_judge(attack, response))
        
        safety_analysis, jailbreak_analysis = await asyncio.gather(safety_task, jailbreak_task)
        
        # Run 5-dimensional BERT MoE (sync but fast — sub-50ms on GPU)
        print("[*] Running 5-Dimensional BERT MoE...")
        threat_vector = self.moe.analyze(attack)
        threat_summary = SpecialistMoE.format_summary(threat_vector)
        print(f"    {threat_summary}")
        
        # Build synthesis prompt with the structured ThreatVector
        synth_prompt = (
            f"Original Attack Prompt:\n{attack}\n\n"
            f"Target Response:\n{response}\n\n"
            f"--- BERT MoE ThreatVector (5-dimensional analysis) ---\n"
            f"{json.dumps(threat_vector, indent=2)}\n\n"
            f"--- LLM Safety Judge Analysis ---\n{safety_analysis}\n\n"
            f"--- LLM Jailbreak Judge Analysis ---\n{jailbreak_analysis}\n"
        )
        
        print("[*] Synthesizing verdicts into JSON...")
        verdict_json = await self.synth_backend.complete_json(synth_prompt, system_prompt=SYNTHESIZER_JUDGE_PROMPT)
        
        # Attach raw MoE vector to the verdict for downstream analysis
        if isinstance(verdict_json, dict):
            verdict_json["_moe_threat_vector"] = threat_vector
        
        return verdict_json
