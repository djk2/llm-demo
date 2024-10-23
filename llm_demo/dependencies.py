from typing import Iterator

from llm_demo.services import BaseLLMService, get_llm_service_cls
from llm_demo.tokenizers import BaseTokenizer, get_tokenizer_cls


def get_tokenizer() -> Iterator[BaseTokenizer]:
    """
    Reuturns a tokenizer instance.
    """
    yield get_tokenizer_cls()()


def get_llm_service() -> Iterator[BaseLLMService]:
    """
    Reuturns a LLM service instance.
    """
    yield get_llm_service_cls()()
