# Import the base ReasoningMode class
from .reasoning_mode import ReasoningMode

class BusinessMode(ReasoningMode):
    """
    Subclass representing the Business preset.
    Pre-configures agents tailored for business ideas, startups, revenue models, and ethical/security risks.
    """
    def __init__(self, backend_override: str = None, model_override: str = None):
        # Business mode instantiates 5 core perspectives:
        # - business: Market fit, competitive landscape, monetization
        # - risk: Financial and operational threats, downsides
        # - ethical: Social impact, fairness, privacy concerns
        # - security: Data vulnerability, IP protection, safety
        # - feasibility: Technical difficulty, resource requirements, time constraints
        super().__init__(
            agent_names=["business", "risk", "ethical", "security", "feasibility"],
            backend_override=backend_override,
            model_override=model_override
        )
