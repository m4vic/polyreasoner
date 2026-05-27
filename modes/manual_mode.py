from typing import List
# Import the base ReasoningMode class
from .reasoning_mode import ReasoningMode

class ManualMode(ReasoningMode):
    """
    Subclass representing the Manual routing mode.
    Allows users to dynamically supply the list of agents they want to run.
    """
    def __init__(self, agent_names: List[str], backend_override: str = None, model_override: str = None):
        # Simply passes the user-specified list of agents directly to the base ReasoningMode
        super().__init__(
            agent_names=agent_names,
            backend_override=backend_override,
            model_override=model_override
        )
