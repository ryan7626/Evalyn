"""Abstract base class for LLM providers.

All LLM providers must implement this interface. This keeps the rest of the
codebase provider-agnostic — swapping Claude for another model is just a
new subclass.
"""

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Abstract LLM interface for Evalyn.

    Provides two completion methods:
    - complete(): returns freeform text
    - complete_json(): returns parsed JSON dict
    """

    @abstractmethod
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int = 4096,
    ) -> str:
        """Generate a text completion.

        Args:
            system_prompt: System-level instructions for the model.
            user_prompt: The user-facing prompt/query.
            max_tokens: Maximum tokens in the response.

        Returns:
            The model's text response.
        """
        ...

    @abstractmethod
    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int = 4096,
    ) -> dict:
        """Generate a structured JSON completion.

        Args:
            system_prompt: System-level instructions. Should instruct the model
                to respond in JSON format.
            user_prompt: The user-facing prompt/query.
            max_tokens: Maximum tokens in the response.

        Returns:
            Parsed JSON dict from the model's response.

        Raises:
            ValueError: If the model's response cannot be parsed as JSON.
        """
        ...
