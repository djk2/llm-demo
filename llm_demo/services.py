import os
import re
from abc import ABC, abstractmethod


class BaseLLMService(ABC):

    @abstractmethod
    def process_prompt(self, prompt: str, text: str) -> str:
        """
        Process the provided text.
        """
        pass


def get_llm_service_cls(name: str | None = None) -> type[BaseLLMService]:
    """
    Reuturns a LLM service class based on the provided name.
    """
    name = name or os.getenv("LLM_SERVICE_CLASS", "mock")

    if name == "mock":
        return MockLLMService
    elif name == "default":
        return DefaultLLMService
    else:
        raise ValueError(f"Unknown LLM service: {name}")


class MockLLMService(BaseLLMService):

    def process_prompt(self, prompt: str, text: str) -> str:
        # Find the token in the text
        pattern = r"<(?:PHONE):[\w-]{36}>"
        matches = re.findall(pattern, text)
        token = matches[0] if matches else ""
        return f"John's phone number is {token}"


class DefaultLLMService(BaseLLMService):

    def process_prompt(self, prompt: str, text: str) -> str:
        raise NotImplementedError("Please provide a LLM service class")
