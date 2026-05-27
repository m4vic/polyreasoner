# Import the base ReasoningMode class
from .reasoning_mode import ReasoningMode

class EvaluateMode(ReasoningMode):
    """
    Subclass representing the Idea Evaluation preset.
    Pre-configures agents tailored for validating product, business, or open-source ideas.
    """
    def __init__(self, backend_override: str = None, model_override: str = None):
        # Idea evaluation mode uses 4 specific business/project lenses:
        # - business: Adoption barriers, monetization risks, competitive environment.
        # - feasibility: Timeline estimation, resource requirements, complexity.
        # - risk: Critical failure modes and dependencies.
        # - contrarian: Skeptical opposition arguing why the idea will fail.
        super().__init__(
            agent_names=["business", "feasibility", "risk", "contrarian"],
            backend_override=backend_override,
            model_override=model_override
        )
