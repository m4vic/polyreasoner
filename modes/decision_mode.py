# Import the base ReasoningMode class
from .reasoning_mode import ReasoningMode

class DecisionMode(ReasoningMode):
    """
    Subclass representing the general Decision preset.
    Pre-configures general-purpose decision-making perspectives.
    """
    def __init__(self, backend_override: str = None, model_override: str = None):
        # Decision mode uses a generic set of 4 analytical lenses:
        # - risk: General risk assessment and threat model.
        # - impact: Scalability, outcomes, and second-order consequences.
        # - ethical: Fairness, moral implications, and stakeholder impact.
        # - contrarian: Direct pushback against the decision to avoid confirmation bias.
        super().__init__(
            agent_names=["risk", "impact", "ethical", "contrarian"],
            backend_override=backend_override,
            model_override=model_override
        )
