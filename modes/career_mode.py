# Import the base ReasoningMode class
from .reasoning_mode import ReasoningMode

class CareerMode(ReasoningMode):
    """
    Subclass representing the Career preset.
    Pre-configures agents tailored for career path changes, new jobs, or promotions.
    """
    def __init__(self, backend_override: str = None, model_override: str = None):
        # Career mode instantiates 4 core perspectives:
        # - feasibility: Is it technically possible to transition/succeed?
        # - risk: What are the primary downsides/threats of this career choice?
        # - impact: What long-term satisfaction or professional scaling does it offer?
        # - contrarian: Arguments against doing it (playing devil's advocate)
        super().__init__(
            agent_names=["feasibility", "risk", "impact", "contrarian"],
            backend_override=backend_override,
            model_override=model_override
        )
