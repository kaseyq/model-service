from abc import ABC, abstractmethod
from typing import Dict, Any

class ModelServiceAdapter(ABC):
    """Abstract base class for model-specific adapters."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.tokenizer = None

    @abstractmethod
    def load(self, timeout: float = 300.0) -> bool:
        """Load the model and tokenizer."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload the model and free resources."""
        pass

    @abstractmethod
    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a prompt request."""
        pass